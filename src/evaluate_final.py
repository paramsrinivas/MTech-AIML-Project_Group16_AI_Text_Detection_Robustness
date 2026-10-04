from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(".")
DATA_PATH = ROOT / "data/hc3_original.csv"
PARAPHRASE_PATH = ROOT / "results/semantic_paraphrases.csv"
OUT = ROOT / "results"
FIG = OUT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

# 1. Load the original HC3 dataset and reconstruct the grouped split.
df = pd.read_csv(DATA_PATH)
required = {"topic_id", "label", "text", "source"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required dataset columns: {missing}")

df = df.dropna(subset=["topic_id", "text", "label"]).copy()
df["topic_id"] = df["topic_id"].astype(str)
df["label"] = df["label"].astype(int)
df["text"] = df["text"].astype(str)

# Keep only questions that have both human and AI answers.
labels_per_topic = df.groupby("topic_id")["label"].agg(
    lambda x: set(x.unique())
)
matched_topics = labels_per_topic[
    labels_per_topic.apply(lambda x: {0, 1}.issubset(x))
].index
df = df[df["topic_id"].isin(matched_topics)].reset_index(drop=True)

splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)
train_idx, test_idx = next(
    splitter.split(df, y=df["label"], groups=df["topic_id"])
)
train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()
train_topics = set(train_df["topic_id"])
test_topics = set(test_df["topic_id"])

assert train_topics.isdisjoint(test_topics)
print(f"Training passages: {len(train_df):,}")
print(f"Training questions: {len(train_topics):,}")
print(f"Held-out questions: {len(test_topics):,}")

# 2. Train the detector on ORIGINAL text only.
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=200_000,
    sublinear_tf=True,
    strip_accents="unicode",
)
X_train = vectorizer.fit_transform(train_df["text"])
model = LinearSVC(class_weight="balanced", random_state=42)
model.fit(X_train, train_df["label"])

# 3. Load the exact same paired topics used for paraphrase generation.
pairs = pd.read_csv(PARAPHRASE_PATH)
needed = {
    "topic_id", "source", "ai_original",
    "ai_surface_edited", "ai_paraphrased", "human_text"
}
missing = needed - set(pairs.columns)
if missing:
    raise ValueError(f"Missing columns in paraphrase file: {missing}")

pairs = pairs.dropna(
    subset=["topic_id", "ai_original", "ai_surface_edited",
            "ai_paraphrased", "human_text"]
).copy()
pairs["topic_id"] = pairs["topic_id"].astype(str)

outside = set(pairs["topic_id"]) - test_topics
if outside:
    raise ValueError(
        f"{len(outside)} paraphrase topics are not in the reconstructed "
        "held-out split. Stop rather than risk data leakage."
    )

# Avoid duplicate questions if the input file contains any.
pairs = pairs.drop_duplicates(subset=["topic_id"]).reset_index(drop=True)
print(f"Paired evaluation questions: {len(pairs)}")

conditions = {
    "Original AI": "ai_original",
    "Surface-edited AI": "ai_surface_edited",
    "Semantic paraphrase": "ai_paraphrased",
}

all_predictions = []
metric_rows = []
source_rows = []
confusions = {}

def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "ai_precision": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "ai_recall": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "ai_f1": f1_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }

# 4. Evaluate the same human controls against each AI-text condition.
for condition_name, ai_column in conditions.items():
    human = pd.DataFrame({
        "topic_id": pairs["topic_id"],
        "source": pairs["source"],
        "label": 0,
        "text": pairs["human_text"].astype(str),
        "text_type": "Human",
    })
    ai = pd.DataFrame({
        "topic_id": pairs["topic_id"],
        "source": pairs["source"],
        "label": 1,
        "text": pairs[ai_column].astype(str),
        "text_type": condition_name,
    })
    eval_df = pd.concat([human, ai], ignore_index=True)
    X_eval = vectorizer.transform(eval_df["text"])
    preds = model.predict(X_eval)

    eval_df["condition"] = condition_name
    eval_df["prediction"] = preds
    eval_df["correct"] = eval_df["label"] == eval_df["prediction"]
    all_predictions.append(eval_df)

    metrics = calculate_metrics(eval_df["label"], preds)
    metrics.update({
        "condition": condition_name,
        "n_passages": len(eval_df),
        "n_human": int((eval_df["label"] == 0).sum()),
        "n_ai": int((eval_df["label"] == 1).sum()),
        "n_questions": eval_df["topic_id"].nunique(),
    })
    metric_rows.append(metrics)
    confusions[condition_name] = [
        [metrics["TN"], metrics["FP"]],
        [metrics["FN"], metrics["TP"]],
    ]

    for source, group in eval_df.groupby("source"):
        sm = calculate_metrics(group["label"], group["prediction"])
        sm.update({
            "condition": condition_name,
            "source": source,
            "n_passages": len(group),
            "n_human": int((group["label"] == 0).sum()),
            "n_ai": int((group["label"] == 1).sum()),
        })
        source_rows.append(sm)

metrics_df = pd.DataFrame(metric_rows)
source_df = pd.DataFrame(source_rows)
predictions_df = pd.concat(all_predictions, ignore_index=True)

metrics_df.to_csv(OUT / "final_condition_metrics.csv", index=False)
source_df.to_csv(OUT / "final_metrics_by_source.csv", index=False)
predictions_df.to_csv(OUT / "final_predictions.csv", index=False)

summary = {
    "model": "TF-IDF word unigrams/bigrams + LinearSVC",
    "training_passages": int(len(train_df)),
    "training_questions": int(len(train_topics)),
    "held_out_questions": int(len(test_topics)),
    "paired_evaluation_questions": int(len(pairs)),
    "human_and_ai_passages_per_condition": int(2 * len(pairs)),
    "train_test_question_overlap": len(train_topics & test_topics),
    "conditions": conditions,
    "confusion_matrices": confusions,
    "metrics": metrics_df.to_dict(orient="records"),
    "note": (
        "Semantic paraphrases are raw model outputs and may contain "
        "meaning changes or factual errors; interpret that condition "
        "as robustness to generated paraphrases, not verified "
        "meaning-preserving paraphrases."
    ),
}
with open(OUT / "final_evaluation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# 5. Comparison chart for the five required metrics.
plot_metrics = [
    ("accuracy", "Accuracy"),
    ("ai_precision", "AI precision"),
    ("ai_recall", "AI recall"),
    ("ai_f1", "AI F1-score"),
    ("false_positive_rate", "False positive rate"),
]
fig, axes = plt.subplots(1, 5, figsize=(18, 4.5))
for ax, (column, title) in zip(axes, plot_metrics):
    ax.bar(metrics_df["condition"], metrics_df[column])
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelrotation=65)
    ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "final_metrics_comparison.png", dpi=180)
plt.close(fig)

# 6. Confusion matrices, with counts printed in each cell.
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, (name, matrix) in zip(axes, confusions.items()):
    ax.imshow(matrix)
    ax.set_title(name)
    ax.set_xticks([0, 1], labels=["Pred Human", "Pred AI"])
    ax.set_yticks([0, 1], labels=["True Human", "True AI"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]),
                    ha="center", va="center")
fig.tight_layout()
fig.savefig(FIG / "final_confusion_matrices.png", dpi=180)
plt.close(fig)

print("\nFINAL EVALUATION COMPLETE")
print(metrics_df[
    ["condition", "n_questions", "accuracy", "ai_precision",
     "ai_recall", "ai_f1", "false_positive_rate", "TN", "FP", "FN", "TP"]
].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nSaved:")
print(" results/final_condition_metrics.csv")
print(" results/final_metrics_by_source.csv")
print(" results/final_predictions.csv")
print(" results/final_evaluation_summary.json")
print(" results/figures/final_metrics_comparison.png")
print(" results/figures/final_confusion_matrices.png")
