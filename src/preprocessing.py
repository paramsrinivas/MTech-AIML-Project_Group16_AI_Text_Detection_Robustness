"""Validation and safe preparation for the AI-text detection experiment."""
from __future__ import annotations
import re
import pandas as pd

REQUIRED_COLUMNS = {
    "passage_id", "topic_id", "condition", "label", "text",
    "parent_id", "provenance", "generator", "edit_method"
}
VALID_CONDITIONS = {"human", "ai_original", "ai_edited"}

def load_and_validate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if df["passage_id"].duplicated().any():
        raise ValueError("passage_id values must be unique.")
    if not set(df["condition"]).issubset(VALID_CONDITIONS):
        raise ValueError(f"condition must be one of {sorted(VALID_CONDITIONS)}")
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    if not set(df["label"]).issubset({0, 1}):
        raise ValueError("label must be 0 (human) or 1 (AI-origin).")
    expected = df["condition"].map({"human": 0, "ai_original": 1, "ai_edited": 1})
    if not (expected == df["label"]).all():
        raise ValueError("Labels conflict with conditions; edited AI must retain label 1.")
    if (df["text"].astype(str).str.strip() == "").any():
        raise ValueError("Empty text passages are not allowed.")
    if df["topic_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Every row must have a topic_id for group-wise splitting.")
    if df["passage_id"].astype(str).str.contains(r"\s", regex=True).any():
        raise ValueError("passage_id values should not contain whitespace.")
    # Normalization is intentionally minimal; retain the original text in the dataset.
    df["text"] = df["text"].astype(str).map(lambda x: re.sub(r"\s+", " ", x).strip())
    return df

def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", str(text)))
