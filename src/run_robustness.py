
from pathlib import Path
import json
import time

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
DATA_PATH = Path("data/hc3_original.csv")
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.20

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)



def make_light_edit(text):
    """
    Apply deterministic, meaning-preserving surface edits.
    This is not a semantic paraphraser.
    """
    import re

    original = str(text)
    edited = original

    # Normalize whitespace.
    edited = re.sub(r"[ \t]+", " ", edited)
    edited = re.sub(r" *\n *", "\n", edited)

    # Apply one deterministic phrase substitution where possible.
    replacements = [
        (r"\bIn addition,\s*", "Additionally, "),
        (r"\bAdditionally,\s*", "Also, "),
        (r"\bFor example,\s*", "For instance, "),
        (r"\bFor instance,\s*", "For example, "),
        (r"\bHowever,\s*", "Nevertheless, "),
        (r"\bTherefore,\s*", "Thus, "),
        (r"\bIn conclusion,\s*", "To conclude, "),
        (r"\bOn the other hand,\s*", "Conversely, "),
    ]

    for pattern, replacement in replacements:
        updated = re.sub(
            pattern,
            replacement,
            edited,
            count=1,
            flags=re.IGNORECASE,
        )
        if updated != edited:
            edited = updated
            break

    edited = re.sub(r"\s+([,.!?;:])", r"\1", edited)
    edited = edited.strip()

    return edited if edited else original


def calculate_metrics(y_true, y_pred):
    """Label 0 = human; label 1 = AI."""
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_ai": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall_ai": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_ai": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def main():
    start = time.time()

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Check that you are in the project folder."
        )

    print("=" * 65)
    print("AI TEXT DETECTION: DETERMINISTIC SURFACE-EDIT TEST")
    print("=" * 65)

    df = pd.read_csv(DATA_PATH)
    required = {"topic_id", "condition", "label", "text", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=list(required)).copy()
    df["topic_id"] = df["topic_id"].astype(str)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    # Match questions that have human and original AI passages.
    conditions = df.groupby("topic_id")["condition"].agg(set)
    matched_ids = conditions[
        conditions.apply(
            lambda values: {"human", "ai_original"}.issubset(values)
        )
    ].index

    df = df[df["topic_id"].isin(matched_ids)].copy()
    df.reset_index(drop=True, inplace=True)

    # Recreate exactly the grouped split used by the baseline script.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_idx, test_idx = next(
        splitter.split(df, y=df["label"], groups=df["topic_id"])
    )
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    overlap = set(train["topic_id"]) & set(test["topic_id"])
    if overlap:
        raise RuntimeError("Train/test question overlap detected.")

    # Train only on original training passages.
    print(f"Training passages: {len(train):,}")
    print(f"Test passages: {len(test):,}")
    print(f"Overlapping question IDs: {len(overlap)}")
    print("\nTraining Linear SVC...")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=200_000,
                sublinear_tf=True,
                strip_accents="unicode",
            ),
        ),
        (
            "classifier",
            LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])
    model.fit(train["text"], train["label"])

    # Use only held-out examples for evaluation.
    human_test = test[test["condition"] == "human"].copy()
    ai_test = test[test["condition"] == "ai_original"].copy()

    if human_test.empty or ai_test.empty:
        raise ValueError(
            "Test split needs both human and ai_original passages."
        )

    # Keep the original AI test passages and create edited counterparts.
    ai_test["edited_text"] = ai_test["text"].map(make_light_edit)

    # Compare the same number of human and AI examples using the available
    # matched held-out questions. All edited passages retain label 1.
    human_eval = human_test[["topic_id", "source", "text", "label"]].copy()
    human_eval["condition_eval"] = "human"

    ai_original_eval = ai_test[
        ["topic_id", "source", "text", "label"]
    ].copy()
    ai_original_eval["condition_eval"] = "ai_original"

    ai_edited_eval = ai_test[
        ["topic_id", "source", "edited_text", "label"]
    ].rename(columns={"edited_text": "text"}).copy()
    ai_edited_eval["condition_eval"] = "ai_edited_surface"

    evaluation = pd.concat(
        [human_eval, ai_original_eval, ai_edited_eval],
        ignore_index=True,
    )

    evaluation["prediction"] = model.predict(evaluation["text"])
    evaluation["correct"] = (
        evaluation["label"] == evaluation["prediction"]
    )

    # Condition-level metrics. FPR is meaningful only for a condition
    # containing human examples; report it as NaN for AI-only conditions.
    rows = []
    for condition_name, group in evaluation.groupby("condition_eval"):
        metrics = calculate_metrics(group["label"], group["prediction"])
        if group["label"].nunique() == 1 and group["label"].iloc[0] == 1:
            metrics["false_positive_rate"] = float("nan")

        metrics["condition"] = condition_name
        metrics["n"] = len(group)
        rows.append(metrics)

    metrics_df = pd.DataFrame(rows)
    metrics_df = metrics_df[
        [
            "condition", "n", "accuracy", "precision_ai",
            "recall_ai", "f1_ai", "false_positive_rate",
            "TN", "FP", "FN", "TP",
        ]
    ]
    metrics_df.to_csv(
        RESULTS_DIR / "robustness_condition_metrics.csv", index=False
    )

    evaluation.to_csv(
        RESULTS_DIR / "robustness_predictions.csv", index=False
    )

    # Paired change in AI detection between each original AI passage and
    # its edited counterpart.
    ai_original_predictions = model.predict(ai_test["text"])
    ai_edited_predictions = model.predict(ai_test["edited_text"])

    paired = pd.DataFrame({
        "topic_id": ai_test["topic_id"].values,
        "source": ai_test["source"].values,
        "original_prediction": ai_original_predictions,
        "edited_prediction": ai_edited_predictions,
        "original_detected_as_ai": ai_original_predictions == 1,
        "edited_detected_as_ai": ai_edited_predictions == 1,
    })
    paired["detection_changed"] = (
        paired["original_prediction"] != paired["edited_prediction"]
    )
    paired.to_csv(
        RESULTS_DIR / "robustness_paired_ai_predictions.csv", index=False
    )

    # Source-level metrics, separated by condition.
    source_rows = []
    for (condition_name, source), group in evaluation.groupby(
        ["condition_eval", "source"]
    ):
        metrics = calculate_metrics(group["label"], group["prediction"])
        if group["label"].nunique() == 1 and group["label"].iloc[0] == 1:
            metrics["false_positive_rate"] = float("nan")
        metrics["condition"] = condition_name
        metrics["source"] = source
        metrics["n"] = len(group)
        source_rows.append(metrics)

    pd.DataFrame(source_rows).to_csv(
        RESULTS_DIR / "robustness_metrics_by_source.csv", index=False
    )

    # Summarize edit impact on AI detection.
    original_detection_rate = paired["original_detected_as_ai"].mean()
    edited_detection_rate = paired["edited_detected_as_ai"].mean()
    changed_rate = paired["detection_changed"].mean()

    summary = {
        "model": "TFIDF_LinearSVC",
        "split_method": "GroupShuffleSplit by topic_id",
        "random_state": RANDOM_STATE,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "overlapping_question_ids": int(len(overlap)),
        "human_test_passages": int(len(human_test)),
        "ai_test_passages": int(len(ai_test)),
        "original_ai_detection_rate": float(original_detection_rate),
        "edited_ai_detection_rate": float(edited_detection_rate),
        "ai_detection_rate_change_percentage_points": float(
            (edited_detection_rate - original_detection_rate) * 100
        ),
        "fraction_of_ai_predictions_changed": float(changed_rate),
        "edit_method": (
            "Deterministic whitespace normalization, punctuation spacing "
            "normalization, and removal of one matching opening discourse "
            "phrase from a small predefined list. No semantic paraphrasing."
        ),
    }

    with open(RESULTS_DIR / "robustness_summary.json", "w") as file:
        json.dump(summary, file, indent=2)

    # Figure 1: report metrics appropriate to each class-specific subset.
    human_rows = metrics_df[metrics_df["condition"] == "human"]
    ai_rows = metrics_df[
        metrics_df["condition"].isin(["ai_original", "ai_edited_surface"])
    ]

    figure, axes = plt.subplots(1, 2, figsize=(11, 5))

    if not human_rows.empty:
        axes[0].bar(
            ["Human passages"],
            human_rows["false_positive_rate"].fillna(0),
        )
    axes[0].set_title("Human False-Positive Rate")
    axes[0].set_ylabel("Fraction incorrectly predicted as AI")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.25)

    if not ai_rows.empty:
        axes[1].bar(
            ai_rows["condition"].map({
                "ai_original": "Original AI",
                "ai_edited_surface": "Surface-edited AI",
            }),
            ai_rows["recall_ai"],
        )
    axes[1].set_title("AI Detection Rate")
    axes[1].set_ylabel("Fraction correctly predicted as AI")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=10)
    axes[1].grid(axis="y", alpha=0.25)

    figure.suptitle("LinearSVC Robustness to Surface Edits")
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "robustness_condition_metrics.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    # Figure 2: AI detection rate before and after edits.
    rates = pd.Series({
        "Original AI": original_detection_rate,
        "Edited AI": edited_detection_rate,
    })
    ax = rates.plot(kind="bar", figsize=(7, 5), ylim=(0, 1))
    ax.set_title("AI Detection Rate Before and After Surface Edits")
    ax.set_ylabel("Fraction predicted as AI")
    ax.set_xlabel("AI passage condition")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "ai_detection_rate_before_after_edits.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print("\n" + "=" * 65)
    print("ROBUSTNESS CONDITION METRICS")
    print("=" * 65)
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nPaired AI detection:")
    print(f"Original AI detection rate: {original_detection_rate:.4f}")
    print(f"Edited AI detection rate:   {edited_detection_rate:.4f}")
    print(f"Prediction changed:         {changed_rate:.4f}")

    print("\nSaved:")
    print("- results/robustness_condition_metrics.csv")
    print("- results/robustness_predictions.csv")
    print("- results/robustness_paired_ai_predictions.csv")
    print("- results/robustness_metrics_by_source.csv")
    print("- results/robustness_summary.json")
    print("- results/figures/robustness_condition_metrics.png")
    print("- results/figures/ai_detection_rate_before_after_edits.png")
    print(f"\nElapsed time: {(time.time() - start) / 60:.2f} min")
    print("ROBUSTNESS EXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()


