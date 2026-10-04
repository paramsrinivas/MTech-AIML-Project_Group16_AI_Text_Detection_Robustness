"""CPU-first, topic-held-out baseline models."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import platform
import sys
import pandas as pd
import sklearn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from src.preprocessing import load_and_validate
from src.evaluation import compute_metrics

def build_models():
    common = dict(ngram_range=(1, 2), min_df=1, max_features=100_000, sublinear_tf=True)
    return {
        "tfidf_logistic_regression": Pipeline([
            ("tfidf", TfidfVectorizer(**common)),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]),
        "tfidf_linear_svm": Pipeline([
            ("tfidf", TfidfVectorizer(**common)),
            ("classifier", LinearSVC(class_weight="balanced", random_state=42)),
        ]),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV using the schema in data/README.md")
    parser.add_argument("--out", default="results")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_and_validate(args.data)
    if df["topic_id"].nunique() < 5:
        raise ValueError("Need at least 5 distinct topics for a minimally useful group holdout; collect more topics.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(df, y=df["label"], groups=df["topic_id"]))
    train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    if train["label"].nunique() < 2 or test["label"].nunique() < 2:
        raise ValueError("The group split did not include both classes in train and test. Add balanced examples across more topics.")
    if set(train["topic_id"]) & set(test["topic_id"]):
        raise RuntimeError("Topic leakage detected in split.")
    split_manifest = {
        "seed": args.seed,
        "test_size": args.test_size,
        "train_topic_ids": sorted(train["topic_id"].unique().tolist()),
        "test_topic_ids": sorted(test["topic_id"].unique().tolist()),
        "n_train": len(train), "n_test": len(test),
        "python": sys.version, "platform": platform.platform(), "scikit_learn": sklearn.__version__,
    }
    (out / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2))

    all_metrics, all_predictions = [], []
    for model_name, model in build_models().items():
        model.fit(train["text"], train["label"])
        pred = model.predict(test["text"])
        pred_frame = test[["passage_id", "topic_id", "condition", "label", "text", "parent_id"]].copy()
        pred_frame["prediction"] = pred
        pred_frame["model"] = model_name
        all_predictions.append(pred_frame)

        for condition in ("ai_original", "ai_edited"):
            subset = test[test["condition"].isin(["human", condition])]
            subset_pred = model.predict(subset["text"])
            all_metrics.append(compute_metrics(
                subset["label"], subset_pred, model_name, condition
            ))

    pd.DataFrame(all_metrics).to_csv(out / "results.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(out / "predictions.csv", index=False)
    print(f"Finished. Results saved to {out.resolve()}")
    print(pd.DataFrame(all_metrics).to_string(index=False))

if __name__ == "__main__":
    main()
