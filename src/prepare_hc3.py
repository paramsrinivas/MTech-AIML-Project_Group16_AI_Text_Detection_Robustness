
"""
Prepare the HC3 English dataset for Assignment 16:
AI Text Detection Robustness.

Outputs:
    data/hc3_original.csv

Conditions:
    human       -> label 0
    ai_original -> label 1

Note:
    This script does not create edited-AI passages. Those will be
    generated and recorded separately in a later experiment stage.
"""

import hashlib
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "hc3_original.csv"

DATASET_NAME = "Hello-SimpleAI/HC3"
DATASET_CONFIG = "default"
DATASET_REVISION = "refs/convert/parquet"

MIN_TEXT_LENGTH = 40


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def clean_text(value):
    """Normalize whitespace without otherwise rewriting the passage."""
    if not isinstance(value, str):
        return ""

    return re.sub(r"\s+", " ", value).strip()


def answer_list(value):
    """Convert an answer field into a list of strings."""
    if isinstance(value, str):
        return [value]

    if isinstance(value, (list, tuple)):
        return list(value)

    return []


def text_hash(text):
    """Return a stable SHA-256 digest of a passage."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Assignment 16: HC3 dataset preparation")
    print("=" * 65)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Config:  {DATASET_CONFIG}")
    print("Loading converted Parquet version...")

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        revision=DATASET_REVISION,
        split="train",
    )

    print("Dataset loaded successfully.")
    print("Columns:", dataset.column_names)
    print("Raw rows:", len(dataset))

    required_columns = {
        "id",
        "question",
        "human_answers",
        "chatgpt_answers",
        "source",
    }

    missing = required_columns - set(dataset.column_names)

    if missing:
        raise ValueError(
            f"Dataset is missing expected columns: {sorted(missing)}\n"
            f"Available columns: {dataset.column_names}"
        )

    # -----------------------------------------------------
    # Convert answers into individual passage records
    # -----------------------------------------------------

    records = []

    for row in dataset:
        question_id = str(row.get("id", "")).strip()
        question = clean_text(row.get("question", ""))
        source = str(row.get("source", "unknown")).strip()

        # Skip rows without a usable question identifier.
        if not question_id:
            continue

        answer_groups = [
            ("human_answers", "human", 0, "human"),
            ("chatgpt_answers", "ai_original", 1, "ChatGPT"),
        ]

        for field, condition, label, generator in answer_groups:
            answers = answer_list(row.get(field))

            for answer_index, answer in enumerate(answers):
                text = clean_text(answer)

                if len(text) < MIN_TEXT_LENGTH:
                    continue

                digest = text_hash(text)

                records.append(
                    {
                        "passage_id": digest[:16],
                        "topic_id": question_id,
                        "question": question,
                        "condition": condition,
                        "label": label,
                        "text": text,
                        "parent_id": "",
                        "provenance": "HC3",
                        "generator": generator,
                        "edit_method": "none",
                        "source": source,
                        "answer_index": answer_index,
                        "text_hash": digest,
                    }
                )

    if not records:
        raise RuntimeError(
            "No passages were extracted. Check the dataset revision "
            "and its schema."
        )

    out = pd.DataFrame(records)

    # -----------------------------------------------------
    # Remove exact duplicate passages
    # -----------------------------------------------------

    before_deduplication = len(out)

    # Preserve the first occurrence of each exact text.
    out = out.drop_duplicates(
        subset=["text_hash"],
        keep="first",
    ).reset_index(drop=True)

    # Verify that the labels and required fields are valid.
    if not set(out["label"].unique()).issubset({0, 1}):
        raise ValueError("Unexpected labels found in the output.")

    if out["text"].isna().any() or (out["text"].str.len() == 0).any():
        raise ValueError("Empty passages remain in the output.")

    # -----------------------------------------------------
    # Save output
    # -----------------------------------------------------

    out.to_csv(OUTPUT_FILE, index=False)

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("DATA PREPARATION COMPLETE")
    print("=" * 65)
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Passages before deduplication: {before_deduplication:,}")
    print(f"Passages after deduplication:  {len(out):,}")
    print(f"Exact duplicates removed:     {before_deduplication - len(out):,}")
    print(f"Distinct question IDs:         {out['topic_id'].nunique():,}")

    print("\nCounts by condition and label:")
    print(
        out.groupby(["condition", "label"])
        .size()
        .to_string()
    )

    print("\nCounts by source and condition:")
    print(
        pd.crosstab(out["source"], out["condition"])
        .to_string()
    )

    print("\nPassage length summary:")
    print(out["text"].str.len().describe().round(1).to_string())

    print("\nNext step: inspect the saved CSV before model training.")


if __name__ == "__main__":
    main()

