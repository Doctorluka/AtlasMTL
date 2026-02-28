from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def evaluate_predictions(
    pred_df: pd.DataFrame,
    true_df: pd.DataFrame,
    label_columns: List[str],
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for col in label_columns:
        pred_col = f"pred_{col}"
        pred = pred_df[pred_col].astype(str)
        true = true_df[col].astype(str)
        covered_mask = pred != "Unknown"

        metrics = {
            "accuracy": float(accuracy_score(true, pred)),
            "macro_f1": float(f1_score(true, pred, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
            "coverage": float(covered_mask.mean()),
        }
        if covered_mask.any():
            covered_true = true[covered_mask]
            covered_pred = pred[covered_mask]
            metrics["covered_accuracy"] = float(accuracy_score(covered_true, covered_pred))
            metrics["risk"] = float(1.0 - metrics["covered_accuracy"])
        else:
            metrics["covered_accuracy"] = 0.0
            metrics["risk"] = 1.0
        results[col] = metrics
    return results
