from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..mapping import closed_loop_unknown_mask, knn_majority_vote, mtl_confidence, mtl_low_conf_mask
from ..utils import progress_iter
from ..version import __version__


def run_model_in_batches(
    model,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
    show_progress: Optional[bool] = None,
) -> tuple[List[torch.Tensor], Dict[str, torch.Tensor], int]:
    logits_batches: Optional[List[List[torch.Tensor]]] = None
    coord_batches: Dict[str, List[torch.Tensor]] = {}
    num_batches = 0

    with torch.no_grad():
        query_loader = DataLoader(torch.tensor(X, dtype=torch.float32), batch_size=batch_size, shuffle=False)
        batch_iterator = progress_iter(
            query_loader,
            total=len(query_loader),
            desc="atlasmtl predict",
            show_progress=show_progress,
        )
        for batch in batch_iterator:
            num_batches += 1
            batch_logits, batch_coords, _ = model.model(batch.to(device))
            if logits_batches is None:
                logits_batches = [[] for _ in range(len(batch_logits))]
            for idx, logit in enumerate(batch_logits):
                logits_batches[idx].append(logit.detach().cpu())
            for name, coord in batch_coords.items():
                coord_batches.setdefault(name, []).append(coord.detach().cpu())

    if logits_batches is None:
        raise ValueError("No cells available for prediction")

    logits = [torch.cat(parts, dim=0) for parts in logits_batches]
    coords_scaled = {name: torch.cat(parts, dim=0) for name, parts in coord_batches.items()}
    return logits, coords_scaled, num_batches


def resolve_knn_space(
    pred_coords: Dict[str, np.ndarray],
    reference_coords: Dict[str, np.ndarray],
) -> tuple[str, Optional[np.ndarray], Optional[np.ndarray]]:
    if "X_pred_latent" in pred_coords and "X_ref_latent" in reference_coords:
        return "latent", pred_coords["X_pred_latent"], reference_coords["X_ref_latent"]
    if "X_pred_umap" in pred_coords and "X_ref_umap" in reference_coords:
        return "umap", pred_coords["X_pred_umap"], reference_coords["X_ref_umap"]
    return "none", None, None


def build_prediction_metadata(
    model,
    knn_correction: str,
    confidence_high: float,
    confidence_low: float,
    margin_threshold: float,
    knn_k: int,
    knn_conf_low: float,
    input_transform: str,
    knn_space_used: str,
    device_used: str,
    device_requested: str,
    num_threads_used: int,
    runtime_summary: Dict[str, object],
) -> Dict[str, object]:
    return {
        "atlasmtl_version": __version__,
        "knn_correction": knn_correction,
        "confidence_high": confidence_high,
        "confidence_low": confidence_low,
        "margin_threshold": margin_threshold,
        "knn_k": knn_k,
        "knn_conf_low": knn_conf_low,
        "latent_source": model.latent_source,
        "input_transform": input_transform,
        "knn_space_used": knn_space_used,
        "device_requested": device_requested,
        "device_used": device_used,
        "num_threads_used": num_threads_used,
        "prediction_runtime": runtime_summary,
        "train_config": model.train_config,
    }


def append_level_predictions(
    pred_df: pd.DataFrame,
    metadata: Dict[str, object],
    *,
    column_name: str,
    probs: np.ndarray,
    label_encoder,
    confidence_high: float,
    confidence_low: float,
    margin_threshold: float,
    knn_correction: str,
    knn_conf_low: float,
    knn_k: int,
    query_space: Optional[np.ndarray],
    ref_space: Optional[np.ndarray],
    ref_labels: np.ndarray,
) -> None:
    max_prob, margin = mtl_confidence(probs)
    raw_idx = probs.argmax(axis=1)
    raw_label = label_encoder.inverse_transform(raw_idx)

    pred_df[f"pred_{column_name}_raw"] = raw_label
    pred_df[f"conf_{column_name}"] = max_prob
    pred_df[f"margin_{column_name}"] = margin

    is_low = mtl_low_conf_mask(max_prob, margin, confidence_high, margin_threshold)
    pred_df[f"is_low_conf_{column_name}"] = is_low

    used_knn = np.zeros_like(is_low, dtype=bool)
    knn_label = raw_label.astype(object).copy()
    knn_vote_frac = np.full(len(raw_label), np.nan, dtype=np.float32)
    is_low_knn_conf = np.zeros_like(is_low, dtype=bool)

    if knn_correction == "all":
        apply_mask = np.ones_like(is_low, dtype=bool)
    elif knn_correction == "low_conf_only":
        apply_mask = is_low.copy()
    else:
        apply_mask = np.zeros_like(is_low, dtype=bool)

    if apply_mask.any() and query_space is not None and ref_space is not None:
        knn_labels, vote_frac, _ = knn_majority_vote(
            ref_coords=ref_space,
            ref_labels=ref_labels,
            query_coords=query_space[apply_mask],
            k=knn_k,
        )
        used_knn[apply_mask] = True
        knn_label[apply_mask] = knn_labels.astype(object)
        knn_vote_frac[apply_mask] = vote_frac
        is_low_knn_conf[apply_mask] = vote_frac < knn_conf_low

    mtl_unknown = max_prob < confidence_low
    is_unknown = closed_loop_unknown_mask(mtl_unknown, is_low, used_knn, is_low_knn_conf)
    final_label = knn_label.astype(object)
    final_label[is_unknown] = "Unknown"

    pred_df[f"pred_{column_name}_knn"] = knn_label
    pred_df[f"knn_vote_frac_{column_name}"] = knn_vote_frac
    pred_df[f"is_low_knn_conf_{column_name}"] = is_low_knn_conf
    pred_df[f"used_knn_{column_name}"] = used_knn
    pred_df[f"is_unknown_{column_name}"] = is_unknown
    pred_df[f"pred_{column_name}"] = final_label
    metadata[f"knn_coverage_{column_name}"] = float(used_knn.mean())
