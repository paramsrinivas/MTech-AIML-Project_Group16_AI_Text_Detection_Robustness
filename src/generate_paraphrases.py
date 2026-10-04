
from pathlib import Path
import re
import time

import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DATA_PATH = Path("data/hc3_original.csv")
OUTPUT_PATH = Path("results/semantic_paraphrases.csv")
MODEL_NAME = "humarin/chatgpt_paraphraser_on_T5_base"

SEED = 42
TEST_SIZE = 0.20
SAMPLE_SIZE = 300
MAX_INPUT_TOKENS = 220


def surface_edit(text):
    text = re.sub(r"[ \t]+", " ", str(text))
    text = re.sub(r" *\n *", "\n", text)

    patterns = [
        r"^\s*Certainly,\s*",
        r"^\s*Sure,\s*",
        r"^\s*Of course,\s*",
        r"^\s*As an AI language model,\s*",
    ]

    for pattern in patterns:
        changed = re.sub(
            pattern, "", text, count=1, flags=re.IGNORECASE
        )
        if changed != text:
            text = changed
            break

    text = re.sub(r"\s+([,.!?;:])", r"\1", text).strip()
    return text or str(text)


def main():
    start = time.time()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = {"topic_id", "condition", "label", "text", "source"}
    if not required.issubset(df.columns):
        raise ValueError(f"Required columns: {sorted(required)}")

    df = df.dropna(subset=list(required)).copy()
    df["topic_id"] = df["topic_id"].astype(str)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    conditions = df.groupby("topic_id")["condition"].agg(set)
    matched_ids = conditions[
        conditions.apply(
            lambda x: {"human", "ai_original"}.issubset(x)
        )
    ].index
    df = df[df["topic_id"].isin(matched_ids)].reset_index(drop=True)

    # Reproduce the baseline's question-grouped split.
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=SEED
    )
    train_idx, test_idx = next(
        splitter.split(df, y=df["label"], groups=df["topic_id"])
    )
    train = df.iloc[train_idx]
    test = df.iloc[test_idx].copy()

    assert not (
        set(train["topic_id"]) & set(test["topic_id"])
    ), "Question leakage detected."

    ai = test[test["condition"] == "ai_original"].copy()
    human = test[test["condition"] == "human"].copy()

    # Use one passage per question and retain only questions with
    # both human and AI passages in the held-out set.
    ai = ai.drop_duplicates("topic_id").set_index("topic_id")
    human = human.drop_duplicates("topic_id").set_index("topic_id")
    paired_ids = sorted(set(ai.index) & set(human.index))

    ai = ai.loc[paired_ids].copy()
    human = human.loc[paired_ids].copy()

    # Limit the pilot sample to texts short enough for this model.
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def count_tokens(text):
        return len(tokenizer(str(text), truncation=False)["input_ids"])

    ai["token_count"] = ai["text"].map(count_tokens)
    ai = ai[ai["token_count"] <= MAX_INPUT_TOKENS].copy()

    if ai.empty:
        raise ValueError("No paired AI passages meet the token limit.")

    n = min(SAMPLE_SIZE, len(ai))
    ai = ai.sample(n=n, random_state=SEED).sort_index()
    human = human.loc[ai.index]

    print(f"Paired eligible questions: {len(ai):,}")
    print(f"Generating paraphrases for: {n:,}")
    print("Loading paraphrasing model...")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    # Resume from a prior partial run when possible.
    existing = {}
    if OUTPUT_PATH.exists():
        old = pd.read_csv(OUTPUT_PATH)
        for _, row in old.iterrows():
            if isinstance(row.get("ai_paraphrased"), str):
                if row["ai_paraphrased"].strip():
                    existing[str(row["topic_id"])] = row.to_dict()

    rows = []
    for i, (topic_id, row) in enumerate(ai.iterrows(), start=1):
        topic_id = str(topic_id)

        if topic_id in existing:
            rows.append(existing[topic_id])
            continue

        original = str(row["text"])
        prompt = "paraphrase: " + original

        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_INPUT_TOKENS,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=192,
                    num_beams=4,
                    do_sample=False,
                    no_repeat_ngram_size=3,
                )

            paraphrase = tokenizer.decode(
                generated[0], skip_special_tokens=True
            ).strip()

            if not paraphrase:
                print(f"Skipping empty output: {topic_id}")
                continue

            human_row = human.loc[topic_id]
            rows.append({
                "topic_id": topic_id,
                "source": str(row["source"]),
                "label": 1,
                "ai_original": original,
                "ai_surface_edited": surface_edit(original),
                "ai_paraphrased": paraphrase,
                "human_text": str(human_row["text"]),
                "human_label": 0,
                "model_name": MODEL_NAME,
            })

        except Exception as exc:
            print(f"Error for question {topic_id}: {exc}")

        # Save progress periodically.
        if i % 10 == 0 or i == len(ai):
            if rows:
                pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
            print(f"Progress: {i}/{len(ai)} candidates")

    if not rows:
        raise RuntimeError("No paraphrases were generated.")

    result = pd.DataFrame(rows).drop_duplicates(
        "topic_id", keep="last"
    )
    result["exact_copy"] = (
        result["ai_original"].str.strip()
        == result["ai_paraphrased"].str.strip()
    )
    result["original_chars"] = result["ai_original"].str.len()
    result["paraphrase_chars"] = result["ai_paraphrased"].str.len()
    result.to_csv(OUTPUT_PATH, index=False)

    print("\nPARAPHRASE GENERATION COMPLETE")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Records: {len(result)}")
    print(f"Exact copies: {int(result['exact_copy'].sum())}")
    print(f"Elapsed: {(time.time() - start) / 60:.2f} minutes")
    print("Manually inspect examples before evaluating robustness.")


if __name__ == "__main__":
    main()

