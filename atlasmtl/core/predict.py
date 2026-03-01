from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from anndata import AnnData

from ..mapping import build_prototypes, openset_score
from ..mapping.hierarchy import enforce_parent_child_consistency
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
    knn_vote_mode: str = "majority",
    knn_reference_mode: str = "full",
    knn_index_mode: str = "exact",
    input_transform: Optional[str] = None,
    apply_calibration: Optional[bool] = None,
    openset_method: Optional[str] = None,
    openset_threshold: Optional[float] = None,
    openset_label_column: Optional[str] = None,
    hierarchy_rules: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
    enforce_hierarchy: bool = False,
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
    knn_vote_mode
        KNN voting strategy. `"majority"` is the default. `"distance_weighted"`
        uses inverse-distance weights for more robust rescue in sparse regions.
    knn_reference_mode
        Reference representation for KNN. `"full"` uses all reference points.
        `"prototypes"` uses per-label centroids (computed per level) to reduce
        reference size.
    knn_index_mode
        Neighbor index implementation. `"exact"` uses scikit-learn exact KNN.
        `"pynndescent"` enables approximate nearest neighbors when installed.
    input_transform
        Optional override for the model's stored input transform. Supported
        values are `"binary"` and `"float"`.
    apply_calibration
        Whether to apply stored post-hoc calibration (e.g. temperature scaling)
        to the model logits before computing probabilities. Defaults to `True`
        when calibration parameters are present on the model, and `False`
        otherwise.
    openset_method
        Optional open-set scoring method used as an additional Unknown signal.
        Supported values are `"nn_distance"` and `"prototype"`. When `None`,
        open-set scoring is disabled (default).
    openset_threshold
        Threshold on the open-set score above which cells are forced to
        `Unknown`. Required when `openset_method` is not `None`.
    openset_label_column
        Label column used to compute prototypes when
        `openset_method="prototype"`. Defaults to the first label column.
    hierarchy_rules
        Optional hierarchy definition used for parent-child consistency checks.
        Format: `{child_col: {"parent_col": <parent>, "child_to_parent": {child: parent}}}`.
    enforce_hierarchy
        Whether to apply hierarchy enforcement after per-level predictions are
        computed. Disabled by default.
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
    if knn_vote_mode not in {"majority", "distance_weighted"}:
        raise ValueError("knn_vote_mode must be one of: majority, distance_weighted")
    if knn_reference_mode not in {"full", "prototypes"}:
        raise ValueError("knn_reference_mode must be one of: full, prototypes")
    if knn_index_mode not in {"exact", "pynndescent"}:
        raise ValueError("knn_index_mode must be one of: exact, pynndescent")
    if openset_method is not None and openset_method not in {"nn_distance", "prototype"}:
        raise ValueError("openset_method must be one of: nn_distance, prototype, or None")
    if openset_method is not None and openset_threshold is None:
        raise ValueError("openset_threshold is required when openset_method is not None")
    if enforce_hierarchy and not hierarchy_rules:
        raise ValueError("hierarchy_rules must be provided when enforce_hierarchy is True")

    resolved_input_transform = input_transform or model.input_transform
    preprocess_metadata = dict(adata.uns.get("atlasmtl_preprocess", {})) if "atlasmtl_preprocess" in adata.uns else None
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

    calibration_payload = None
    if isinstance(getattr(model, "train_config", None), dict):
        calibration_payload = model.train_config.get("calibration")
    has_calibration = bool(calibration_payload and calibration_payload.get("method") == "temperature_scaling")
    should_apply_calibration = has_calibration if apply_calibration is None else bool(apply_calibration)
    calibration_temperatures = None
    if should_apply_calibration and has_calibration:
        temps = calibration_payload.get("temperatures", {})
        calibration_temperatures = {k: float(v) for k, v in temps.items()}
        for i, col in enumerate(model.label_columns):
            temp = calibration_temperatures.get(col)
            if temp and temp > 0:
                logits[i] = logits[i] / float(temp)

    pred_coords: Dict[str, np.ndarray] = {}
    for name, coord in coords_scaled.items():
        pred_coords[f"X_pred_{name}"] = unscale_coords(coord.numpy(), model.coord_stats[name])

    knn_space_used, query_space, ref_space = resolve_knn_space(pred_coords, model.reference_coords)

    openset_scores = None
    openset_unknown = None
    openset_num_prototypes = None
    openset_space_used = None
    if openset_method is not None and query_space is not None and ref_space is not None:
        openset_space_used = knn_space_used
        ref_labels = None
        resolved_label_column = openset_label_column or (model.label_columns[0] if model.label_columns else None)
        if openset_method == "prototype":
            if not resolved_label_column:
                raise ValueError("openset_label_column must be provided when no label columns are available")
            ref_labels = model.reference_labels.get(resolved_label_column)
            if ref_labels is None:
                raise ValueError(f"Missing reference labels for openset_label_column={resolved_label_column}")
        openset_scores, openset_num_prototypes = openset_score(
            ref_space,
            query_space,
            method=openset_method,
            ref_labels=ref_labels,
        )
        threshold = float(openset_threshold)
        openset_unknown = openset_scores > threshold

    metadata = build_prediction_metadata(
        model=model,
        knn_correction=knn_correction,
        confidence_high=confidence_high,
        confidence_low=confidence_low,
        margin_threshold=margin_threshold,
        knn_k=knn_k,
        knn_conf_low=knn_conf_low,
        input_transform=resolved_input_transform,
        calibration_applied=bool(should_apply_calibration and has_calibration),
        calibration_method=(calibration_payload.get("method") if has_calibration else None),
        calibration_temperatures=calibration_temperatures,
        knn_space_used=knn_space_used,
        device_used=resolved_device.type,
        device_requested=device,
        num_threads_used=num_threads_used,
        runtime_summary=prediction_runtime,
        preprocess_metadata=preprocess_metadata,
    )
    metadata["knn_vote_mode"] = knn_vote_mode
    metadata["knn_reference_mode"] = knn_reference_mode
    metadata["knn_index_mode"] = knn_index_mode
    if openset_method is not None:
        metadata["openset_method"] = openset_method
        metadata["openset_threshold"] = float(openset_threshold) if openset_threshold is not None else None
        metadata["openset_space_used"] = openset_space_used
        metadata["openset_label_column"] = openset_label_column
        if openset_unknown is not None:
            metadata["openset_unknown_rate"] = float(openset_unknown.mean())
        if openset_num_prototypes is not None:
            metadata["openset_num_prototypes"] = int(openset_num_prototypes)

    for i, col in enumerate(model.label_columns):
        probs = torch.softmax(logits[i], dim=1).numpy()
        ref_labels_for_col = model.reference_labels[col]
        ref_space_for_col = ref_space
        if knn_reference_mode == "prototypes" and ref_space is not None:
            proto_coords, proto_labels = build_prototypes(ref_space, ref_labels_for_col)
            ref_space_for_col = proto_coords
            ref_labels_for_col = proto_labels
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
            knn_vote_mode=knn_vote_mode,
            knn_index_mode=knn_index_mode,
            query_space=query_space,
            ref_space=ref_space_for_col,
            ref_labels=ref_labels_for_col,
            openset_unknown=openset_unknown,
        )

    if enforce_hierarchy and hierarchy_rules:
        enforced_rates = {}
        for child_col, rule in hierarchy_rules.items():
            parent_col = rule.get("parent_col")
            child_to_parent = rule.get("child_to_parent")
            if not parent_col or not isinstance(child_to_parent, dict):
                raise ValueError("hierarchy_rules entries must contain parent_col and child_to_parent mapping")
            pred_df, rate = enforce_parent_child_consistency(
                pred_df,
                parent_col=str(parent_col),
                child_col=str(child_col),
                child_to_parent={str(k): str(v) for k, v in child_to_parent.items()},
            )
            enforced_rates[str(child_col)] = float(rate)
        metadata["hierarchy_enforced"] = True
        metadata["hierarchy_inconsistency_rate"] = enforced_rates

    result = PredictionResult(predictions=pred_df, coordinates=pred_coords, metadata=metadata)
    if resolve_show_summary(show_summary):
        result.show_resource_usage()
    return result
