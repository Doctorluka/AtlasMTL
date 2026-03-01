from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.manifold import trustworthiness
from ..mapping import hierarchy_path_consistency_rate


def _ece_score(confidence: np.ndarray, correct: np.ndarray, *, n_bins: int = 15) -> float:
    if confidence.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(confidence, bins, right=True) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        acc = float(correct[mask].mean())
        conf = float(confidence[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(acc - conf)
    return float(ece)


def _brier_score(confidence: np.ndarray, correct: np.ndarray) -> float:
    if confidence.size == 0:
        return 0.0
    return float(np.mean((confidence - correct) ** 2))


def _aurc_score(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve for selective classification.

    Here coverage is increased by accepting higher-confidence items first.
    """
    n = int(confidence.size)
    if n == 0:
        return 0.0
    order = np.argsort(-confidence)
    correct_sorted = correct[order].astype(np.float32, copy=False)
    cum_errors = np.cumsum(1.0 - correct_sorted)
    coverage = np.arange(1, n + 1, dtype=np.float32) / float(n)
    risk = cum_errors / np.arange(1, n + 1, dtype=np.float32)
    # NumPy 2.0 removed np.trapz; use trapezoid integration.
    return float(np.trapezoid(risk, coverage))


def _conf_and_correct(
    pred_df: pd.DataFrame, true_df: pd.DataFrame, *, col: str
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    conf_col = f"conf_{col}"
    pred_col = f"pred_{col}"
    if conf_col not in pred_df.columns or pred_col not in pred_df.columns or col not in true_df.columns:
        return None
    pred = pred_df[pred_col].astype(str)
    true = true_df[col].astype(str)
    covered_mask = (pred != "Unknown").to_numpy()
    conf = pred_df[conf_col].to_numpy(dtype=np.float32, copy=False)
    correct = (pred == true).to_numpy(dtype=bool, copy=False)
    return conf, correct.astype(np.float32), covered_mask


def evaluate_predictions(
    pred_df: pd.DataFrame,
    true_df: pd.DataFrame,
    label_columns: List[str],
    *,
    n_bins: int = 15,
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
            "reject_rate": float(1.0 - covered_mask.mean()),
            "n_total": float(len(true)),
            "n_covered": float(int(covered_mask.sum())),
        }
        if covered_mask.any():
            covered_true = true[covered_mask]
            covered_pred = pred[covered_mask]
            metrics["covered_accuracy"] = float(accuracy_score(covered_true, covered_pred))
            metrics["risk"] = float(1.0 - metrics["covered_accuracy"])
        else:
            metrics["covered_accuracy"] = 0.0
            metrics["risk"] = 1.0

        conf_bundle = _conf_and_correct(pred_df, true_df, col=col)
        if conf_bundle is not None:
            conf, correct, covered = conf_bundle
            conf_cov = conf[covered]
            correct_cov = correct[covered]
            metrics["ece"] = _ece_score(conf_cov, correct_cov, n_bins=n_bins)
            metrics["brier"] = _brier_score(conf_cov, correct_cov)
            metrics["aurc"] = _aurc_score(conf_cov, correct_cov)
        results[col] = metrics
    return results


def evaluate_predictions_by_group(
    pred_df: pd.DataFrame,
    true_df: pd.DataFrame,
    label_columns: List[str],
    *,
    group: pd.Series,
    n_bins: int = 15,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Evaluate metrics per group value.

    Returns: {group_value: {label_column: metrics_dict}}
    """
    if len(group) != len(pred_df):
        raise ValueError("group must have the same length as pred_df")
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for g_value, idx in group.astype(str).groupby(group.astype(str)).groups.items():
        sel_pred = pred_df.loc[idx]
        sel_true = true_df.loc[idx]
        out[str(g_value)] = evaluate_predictions(sel_pred, sel_true, label_columns, n_bins=n_bins)
    return out


def evaluate_prediction_behavior(
    pred_df: pd.DataFrame,
    true_df: pd.DataFrame,
    label_columns: List[str],
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for col in label_columns:
        pred_col = f"pred_{col}"
        raw_col = f"pred_{col}_raw"
        knn_col = f"pred_{col}_knn"
        used_knn_col = f"used_knn_{col}"
        unknown_col = f"is_unknown_{col}"

        metrics: Dict[str, float] = {}
        n_obs = float(len(pred_df))
        metrics["unknown_rate"] = (
            float(pred_df[unknown_col].to_numpy(dtype=bool, copy=False).mean()) if unknown_col in pred_df.columns else 0.0
        )
        metrics["knn_coverage"] = (
            float(pred_df[used_knn_col].to_numpy(dtype=bool, copy=False).mean()) if used_knn_col in pred_df.columns else 0.0
        )

        if used_knn_col in pred_df.columns and raw_col in pred_df.columns and knn_col in pred_df.columns:
            used_knn = pred_df[used_knn_col].to_numpy(dtype=bool, copy=False)
            raw_pred = pred_df[raw_col].astype(str).to_numpy()
            knn_pred = pred_df[knn_col].astype(str).to_numpy()
            final_pred = pred_df[pred_col].astype(str).to_numpy() if pred_col in pred_df.columns else knn_pred
            changed = used_knn & (raw_pred != knn_pred)
            metrics["knn_change_rate"] = float(changed.mean()) if n_obs else 0.0
            metrics["knn_change_rate_among_used"] = float(changed[used_knn].mean()) if used_knn.any() else 0.0
            if col in true_df.columns:
                true = true_df[col].astype(str).to_numpy()
                rescued = used_knn & (raw_pred != true) & (final_pred == true)
                harmed = used_knn & (raw_pred == true) & (final_pred != true)
                metrics["knn_rescue_rate"] = float(rescued.mean()) if n_obs else 0.0
                metrics["knn_rescue_rate_among_used"] = float(rescued[used_knn].mean()) if used_knn.any() else 0.0
                metrics["knn_harm_rate"] = float(harmed.mean()) if n_obs else 0.0
                metrics["knn_harm_rate_among_used"] = float(harmed[used_knn].mean()) if used_knn.any() else 0.0
        results[col] = metrics
    return results


def evaluate_prediction_behavior_by_group(
    pred_df: pd.DataFrame,
    true_df: pd.DataFrame,
    label_columns: List[str],
    *,
    group: pd.Series,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    if len(group) != len(pred_df):
        raise ValueError("group must have the same length as pred_df")
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for g_value, idx in group.astype(str).groupby(group.astype(str)).groups.items():
        sel_pred = pred_df.loc[idx]
        sel_true = true_df.loc[idx]
        out[str(g_value)] = evaluate_prediction_behavior(sel_pred, sel_true, label_columns)
    return out


def evaluate_hierarchy_metrics(
    pred_df: pd.DataFrame,
    true_df: pd.DataFrame,
    label_columns: List[str],
    *,
    hierarchy_rules: Dict[str, Dict[str, Dict[str, str]]],
    unknown_label: str = "Unknown",
) -> Dict[str, object]:
    metrics: Dict[str, object] = {"edges": {}}
    if not hierarchy_rules:
        return metrics

    full_path_cols = [col for col in label_columns if f"pred_{col}" in pred_df.columns and col in true_df.columns]
    if full_path_cols:
        pred_path = pred_df[[f"pred_{col}" for col in full_path_cols]].astype(str)
        true_path = true_df[full_path_cols].astype(str)
        exact = (pred_path.to_numpy() == true_path.to_numpy()).all(axis=1)
        covered = ~(pred_path == unknown_label).any(axis=1).to_numpy()
        metrics["full_path_accuracy"] = float(exact.mean()) if len(exact) else 0.0
        metrics["full_path_coverage"] = float(covered.mean()) if len(covered) else 0.0
        metrics["full_path_covered_accuracy"] = float(exact[covered].mean()) if covered.any() else 0.0

    for child_col, rule in hierarchy_rules.items():
        parent_col = rule.get("parent_col")
        child_to_parent = rule.get("child_to_parent")
        if not parent_col or not isinstance(child_to_parent, dict):
            raise ValueError("hierarchy_rules entries must contain parent_col and child_to_parent mapping")
        metrics["edges"][str(child_col)] = {
            "parent_col": str(parent_col),
            "path_consistency_rate": hierarchy_path_consistency_rate(
                pred_df,
                parent_col=str(parent_col),
                child_col=str(child_col),
                child_to_parent={str(k): str(v) for k, v in child_to_parent.items()},
                unknown_label=unknown_label,
            ),
        }
    return metrics


def _neighbor_overlap_score(true_coords: np.ndarray, pred_coords: np.ndarray, *, n_neighbors: int) -> float:
    n_obs = int(true_coords.shape[0])
    if n_obs <= 1:
        return 0.0
    k_eff = min(int(n_neighbors), n_obs - 1)
    if k_eff <= 0:
        return 0.0
    true_dist = np.linalg.norm(true_coords[:, None, :] - true_coords[None, :, :], axis=2)
    pred_dist = np.linalg.norm(pred_coords[:, None, :] - pred_coords[None, :, :], axis=2)
    np.fill_diagonal(true_dist, np.inf)
    np.fill_diagonal(pred_dist, np.inf)
    true_idx = np.argsort(true_dist, axis=1)[:, :k_eff]
    pred_idx = np.argsort(pred_dist, axis=1)[:, :k_eff]
    overlaps = []
    for true_row, pred_row in zip(true_idx, pred_idx):
        overlaps.append(len(set(true_row.tolist()) & set(pred_row.tolist())) / float(k_eff))
    return float(np.mean(overlaps)) if overlaps else 0.0


def evaluate_coordinate_metrics(
    coordinates: Dict[str, np.ndarray],
    true_coordinates: Dict[str, np.ndarray],
    *,
    n_neighbors: int = 10,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for name, pred_arr in coordinates.items():
        if name not in true_coordinates:
            continue
        pred = np.asarray(pred_arr, dtype=np.float32)
        true = np.asarray(true_coordinates[name], dtype=np.float32)
        if pred.shape != true.shape:
            raise ValueError(f"coordinate shapes must match for {name}: {pred.shape} vs {true.shape}")
        metrics[f"{name}_rmse"] = float(np.sqrt(np.mean((pred - true) ** 2)))
        if pred.shape[0] <= 1:
            metrics[f"{name}_trustworthiness"] = 0.0
            metrics[f"{name}_continuity"] = 0.0
            metrics[f"{name}_neighbor_overlap"] = 0.0
            continue
        k_eff = min(int(n_neighbors), pred.shape[0] - 1)
        max_trust_k = max((pred.shape[0] // 2) - 1, 0)
        trust_k = min(k_eff, max_trust_k)
        if trust_k <= 0:
            metrics[f"{name}_trustworthiness"] = 0.0
            metrics[f"{name}_continuity"] = 0.0
            metrics[f"{name}_neighbor_overlap"] = _neighbor_overlap_score(true, pred, n_neighbors=k_eff)
            continue
        metrics[f"{name}_trustworthiness"] = float(trustworthiness(true, pred, n_neighbors=trust_k))
        metrics[f"{name}_continuity"] = float(trustworthiness(pred, true, n_neighbors=trust_k))
        metrics[f"{name}_neighbor_overlap"] = _neighbor_overlap_score(true, pred, n_neighbors=k_eff)
    return metrics
