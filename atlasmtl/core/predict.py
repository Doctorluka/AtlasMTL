from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from anndata import AnnData

from .data import extract_matrix, unscale_coords
from .predict_utils import (
    append_level_predictions,
    build_prediction_metadata,
    resolve_knn_space,
    run_model_in_batches,
)
from .runtime import configure_torch_threads, resolve_device
from .types import PredictionResult, TrainedModel
from ..utils import RuntimeMonitor, resolve_show_summary


def predict(
    model: TrainedModel,
    adata: AnnData,
    knn_correction: str = "low_conf_only",
    confidence_high: float = 0.7,
    confidence_low: float = 0.4,
    margin_threshold: float = 0.2,
    knn_k: int = 15,
    knn_conf_low: float = 0.6,
    input_transform: Optional[str] = None,
    batch_size: int = 256,
    num_threads: Union[int, str, None] = 10,
    device: str = "auto",
    show_progress: Optional[bool] = None,
    show_summary: Optional[bool] = None,
) -> PredictionResult:
    """Run annotation and optional KNN correction on a query `AnnData`.

    Parameters
    ----------
    model
        A trained `TrainedModel` returned by `build_model()` or loaded from disk.
    adata
        Query dataset. Genes are aligned to the model's training genes before
        inference.
    knn_correction
        KNN usage mode. `"off"` disables KNN entirely, `"low_conf_only"`
        applies KNN only to low-confidence predictions, and `"all"` applies KNN
        to every cell.
    confidence_high
        Upper confidence threshold used together with `margin_threshold` to
        decide whether a prediction is considered low-confidence and should be
        routed to KNN when `knn_correction="low_conf_only"`.
    confidence_low
        Lower confidence threshold used in the closed-loop Unknown policy.
        Predictions below this threshold can be marked `Unknown`.
    margin_threshold
        Minimum difference between the top-1 and top-2 class probabilities for
        a prediction to be considered well separated.
    knn_k
        Number of reference neighbors used for KNN majority voting.
    knn_conf_low
        Minimum KNN vote fraction required for a KNN-corrected prediction to
        avoid `Unknown` when MTL confidence is already low.
    input_transform
        Optional override for the model's stored input transform. Supported
        values are `"binary"` and `"float"`.
    batch_size
        Inference batch size.
    num_threads
        Number of CPU threads made available to PyTorch during inference.
        Default is `10`. Pass `"max"` to use up to 80% of available CPUs.
    device
        Execution device: `"auto"`, `"cpu"`, or `"cuda"`.
    show_progress
        Whether to display an inference progress bar with ETA. Defaults to
        auto-detection based on whether stderr is attached to a terminal.
    show_summary
        Whether to print a compact post-inference resource summary. Defaults to
        auto-detection based on whether stdout is attached to a terminal.

    Returns
    -------
    PredictionResult
        In-memory prediction bundle. Use `to_adata()`, `to_dataframe()`, or
        `to_csv()` to export selected outputs.
    """
    if knn_correction not in {"off", "low_conf_only", "all"}:
        raise ValueError("knn_correction must be one of: off, low_conf_only, all")
    if knn_correction != "off" and not model.reference_coords:
        raise ValueError(
            "KNN correction requires reference data. Load a model with reference storage enabled "
            "or run predict(knn_correction='off')."
        )

    resolved_input_transform = input_transform or model.input_transform
    X = extract_matrix(adata, model.train_genes, input_transform=resolved_input_transform)
    num_threads_used = configure_torch_threads(num_threads)
    resolved_device = resolve_device(device)
    model.model.to(resolved_device)
    model.model.eval()

    runtime_monitor = RuntimeMonitor(phase="predict", device=resolved_device)
    logits, coords_scaled, num_batches = run_model_in_batches(
        model,
        X,
        batch_size,
        resolved_device,
        show_progress=show_progress,
    )
    prediction_runtime = runtime_monitor.finish(num_items=adata.n_obs, num_batches=num_batches)
    pred_df = pd.DataFrame(index=adata.obs_names)

    pred_coords: Dict[str, np.ndarray] = {}
    for name, coord in coords_scaled.items():
        pred_coords[f"X_pred_{name}"] = unscale_coords(coord.numpy(), model.coord_stats[name])

    knn_space_used, query_space, ref_space = resolve_knn_space(pred_coords, model.reference_coords)
    metadata = build_prediction_metadata(
        model,
        knn_correction,
        confidence_high,
        confidence_low,
        margin_threshold,
        knn_k,
        knn_conf_low,
        resolved_input_transform,
        knn_space_used,
        resolved_device.type,
        device,
        num_threads_used,
        prediction_runtime,
    )

    for i, col in enumerate(model.label_columns):
        probs = torch.softmax(logits[i], dim=1).numpy()
        append_level_predictions(
            pred_df,
            metadata,
            column_name=col,
            probs=probs,
            label_encoder=model.label_encoders[col],
            confidence_high=confidence_high,
            confidence_low=confidence_low,
            margin_threshold=margin_threshold,
            knn_correction=knn_correction,
            knn_conf_low=knn_conf_low,
            knn_k=knn_k,
            query_space=query_space,
            ref_space=ref_space,
            ref_labels=model.reference_labels[col],
        )

    result = PredictionResult(predictions=pred_df, coordinates=pred_coords, metadata=metadata)
    if resolve_show_summary(show_summary):
        result.show_resource_usage()
    return result
