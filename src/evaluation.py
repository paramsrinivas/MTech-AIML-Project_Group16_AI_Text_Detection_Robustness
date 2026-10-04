"""Evaluation metrics for binary AI-origin detection."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def compute_metrics(y_true, y_pred, model_name="", condition_subset=""):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) else float("nan")
    return {
        "model": model_name,
        "evaluation_subset": condition_subset,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_ai": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_ai": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_ai": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": fpr,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

def summarize_by_condition(df: pd.DataFrame, pred_col="prediction"):
    rows = []
    for condition in ("ai_original", "ai_edited"):
        subset = df[df["condition"].isin(["human", condition])]
        rows.append(compute_metrics(
            subset["label"], subset[pred_col],
            model_name=str(df["model"].iloc[0]) if "model" in df else "",
            condition_subset=condition
        ))
    return pd.DataFrame(rows)
