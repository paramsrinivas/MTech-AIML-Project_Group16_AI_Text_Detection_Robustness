
from pathlib import Path
import json
import time

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------
DATA_PATH = Path("data/hc3_original.csv")
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
RANDOM_STATE = 42
TEST_SIZE = 0.20

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true, y_pred):
    """Calculate AI-positive metrics; label 1 = AI, label 0 = human."""
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
    start_time = time.time()

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run src/prepare_hc3.py first."
        )

    print("=" * 65)
    print("AI TEXT DETECTION ROBUSTNESS: BASELINE EXPERIMENT")
    print("=" * 65)

    df = pd.read_csv(DATA_PATH)
    required = {"topic_id", "condition", "label", "text", "source"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    df = df.dropna(subset=list(required)).copy()
    df["topic_id"] = df["topic_id"].astype(str)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    # Keep questions that have both human and original AI answers.
    conditions = df.groupby("topic_id")["condition"].agg(set)
    matched_ids = conditions[
        conditions.apply(
            lambda values: {"human", "ai_original"}.issubset(values)
        )
    ].index

    df = df[df["topic_id"].isin(matched_ids)].copy()
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No matched questions found. Check topic_id and condition values."
        )

    print(f"Passages after matching questions: {len(df):,}")
    print(f"Distinct matched questions: {df['topic_id'].nunique():,}")
    print("\nLabel counts (0=human, 1=AI):")
    print(df["label"].value_counts().sort_index().to_string())

    # Split by question ID so passages from one question cannot appear
    # in both train and test sets.
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

    manifest = {
        "dataset": "Hello-SimpleAI/HC3",
        "input_file": str(DATA_PATH),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "split_method": "GroupShuffleSplit by topic_id",
        "train_rows": len(train),
        "test_rows": len(test),
        "train_questions": int(train["topic_id"].nunique()),
        "test_questions": int(test["topic_id"].nunique()),
        "overlapping_question_ids": len(
            set(train["topic_id"]) & set(test["topic_id"])
        ),
        "matched_question_count": int(df["topic_id"].nunique()),
    }

    with open(RESULTS_DIR / "split_manifest.json", "w") as file:
        json.dump(manifest, file, indent=2)

    print("\nSplit summary:")
    print(json.dumps(manifest, indent=2))

    if manifest["overlapping_question_ids"] != 0:
        raise RuntimeError("Data leakage detected: question IDs overlap.")

    X_train = train["text"]
    y_train = train["label"]
    X_test = test["text"]
    y_test = test["label"]

    # Two lightweight classical baselines suitable for a MacBook.
    models = {
        "TFIDF_LogisticRegression": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=200_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    dtype=float,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
        "TFIDF_LinearSVC": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=200_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    dtype=float,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
    }

    all_metrics = []
    all_predictions = []
    all_source_metrics = []

    for model_name, model in models.items():
        print("\n" + "-" * 65)
        print(f"Training: {model_name}")
        model_start = time.time()

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = calculate_metrics(y_test, predictions)
        metrics["model"] = model_name
        all_metrics.append(metrics)

        print(f"Elapsed time: {(time.time() - model_start) / 60:.2f} min")
        print(f"Accuracy:           {metrics['accuracy']:.4f}")
        print(f"AI precision:       {metrics['precision_ai']:.4f}")
        print(f"AI recall:          {metrics['recall_ai']:.4f}")
        print(f"AI F1:              {metrics['f1_ai']:.4f}")
        print(f"False positive rate:{metrics['false_positive_rate']:.4f}")
        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["Human", "AI"],
                zero_division=0,
            )
        )

        # Store predictions for later robustness analysis.
        prediction_frame = test[
            ["topic_id", "condition", "label", "source", "text"]
        ].copy()
        prediction_frame["model"] = model_name
        prediction_frame["prediction"] = predictions
        prediction_frame["correct"] = (
            prediction_frame["label"] == prediction_frame["prediction"]
        )
        all_predictions.append(prediction_frame)

        # Evaluate the test set separately for each source.
        for source, group in prediction_frame.groupby("source"):
            source_metrics = calculate_metrics(
                group["label"], group["prediction"]
            )
            source_metrics["model"] = model_name
            source_metrics["source"] = source
            source_metrics["n"] = len(group)
            all_source_metrics.append(source_metrics)

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    source_df = pd.DataFrame(all_source_metrics)

    metrics_df.to_csv(RESULTS_DIR / "baseline_results.csv", index=False)
    predictions_df.to_csv(
        RESULTS_DIR / "baseline_predictions.csv", index=False
    )
    source_df.to_csv(
        RESULTS_DIR / "baseline_metrics_by_source.csv", index=False
    )

    # Plot headline metrics for the two models.
    metric_columns = [
        "accuracy",
        "precision_ai",
        "recall_ai",
        "f1_ai",
        "false_positive_rate",
    ]
    plot_df = metrics_df.set_index("model")[metric_columns]

    ax = plot_df.plot(kind="bar", figsize=(12, 6), ylim=(0, 1))
    ax.set_title("HC3 Baseline Text Detection Metrics")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "baseline_metrics.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print("\n" + "=" * 65)
    print("BASELINE RESULTS")
    print("=" * 65)
    print(
        metrics_df[
            [
                "model",
                "accuracy",
                "precision_ai",
                "recall_ai",
                "f1_ai",
                "false_positive_rate",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )

    print("\nSaved files:")
    print(f"- {RESULTS_DIR / 'baseline_results.csv'}")
    print(f"- {RESULTS_DIR / 'baseline_predictions.csv'}")
    print(f"- {RESULTS_DIR / 'baseline_metrics_by_source.csv'}")
    print(f"- {RESULTS_DIR / 'split_manifest.json'}")
    print(f"- {FIGURES_DIR / 'baseline_metrics.png'}")
    print(f"\nTotal elapsed time: {(time.time() - start_time) / 60:.2f} min")
    print("BASELINE EXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()

