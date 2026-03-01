from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def enforce_parent_child_consistency(
    pred_df: pd.DataFrame,
    *,
    parent_col: str,
    child_col: str,
    child_to_parent: Dict[str, str],
    unknown_label: str = "Unknown",
) -> Tuple[pd.DataFrame, float]:
    """Force child predictions to be consistent with parent predictions.

    If `pred_child` maps to a different parent than `pred_parent`, the child is
    set to Unknown and marked as unknown.
    """
    parent_pred_col = f"pred_{parent_col}"
    child_pred_col = f"pred_{child_col}"
    child_unknown_col = f"is_unknown_{child_col}"
    if parent_pred_col not in pred_df.columns or child_pred_col not in pred_df.columns:
        raise ValueError("pred_df is missing required pred_* columns for hierarchy enforcement")

    parent = pred_df[parent_pred_col].astype(str).to_numpy()
    child = pred_df[child_pred_col].astype(str).to_numpy()
    mapped_parent = np.array([child_to_parent.get(x) for x in child], dtype=object)
    inconsistent = (mapped_parent != parent) & (child != unknown_label) & (parent != unknown_label) & (mapped_parent != None)
    if child_unknown_col in pred_df.columns:
        pred_df[child_unknown_col] = pred_df[child_unknown_col].to_numpy(dtype=bool) | inconsistent
    else:
        pred_df[child_unknown_col] = inconsistent
    child_new = child.astype(object)
    child_new[inconsistent] = unknown_label
    pred_df[child_pred_col] = child_new
    rate = float(inconsistent.mean()) if len(inconsistent) else 0.0
    return pred_df, rate


def hierarchy_path_consistency_rate(
    pred_df: pd.DataFrame,
    *,
    parent_col: str,
    child_col: str,
    child_to_parent: Dict[str, str],
    unknown_label: str = "Unknown",
) -> float:
    parent = pred_df[f"pred_{parent_col}"].astype(str).to_numpy()
    child = pred_df[f"pred_{child_col}"].astype(str).to_numpy()
    covered = (parent != unknown_label) & (child != unknown_label)
    if not covered.any():
        return 0.0
    mapped_parent = np.array([child_to_parent.get(x) for x in child], dtype=object)
    ok = mapped_parent == parent
    ok = ok & covered
    return float(ok.sum() / covered.sum())

