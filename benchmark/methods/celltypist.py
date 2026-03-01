from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd
from anndata import read_h5ad

from atlasmtl.core.evaluate import (
    evaluate_prediction_behavior,
    evaluate_prediction_behavior_by_group,
    evaluate_predictions,
    evaluate_predictions_by_group,
)
from atlasmtl.models.checksums import artifact_checksums


def _runtime_payload(*, phase: str, elapsed_seconds: float, n_items: int) -> Dict[str, object]:
    throughput = float(n_items / elapsed_seconds) if elapsed_seconds > 0 else None
    return {
        "phase": phase,
        "elapsed_seconds": float(elapsed_seconds),
        "items_per_second": throughput,
        "process_peak_rss_gb": None,
        "gpu_peak_memory_gb": None,
    }


def _resolve_path(value: str, *, manifest_path: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((manifest_path.parent / path).resolve())


def _top2_margin(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.shape[1] <= 1:
        return np.zeros(probabilities.shape[0], dtype=np.float32)
    sorted_probs = np.sort(probabilities, axis=1)
    return (sorted_probs[:, -1] - sorted_probs[:, -2]).astype(np.float32)


def _resolve_model_map(
    method_cfg: Mapping[str, Any],
    *,
    label_columns: Iterable[str],
    manifest_path: Path,
) -> Dict[str, str]:
    label_columns = list(label_columns)
    if "models" in method_cfg:
        models_cfg = method_cfg["models"]
        if not isinstance(models_cfg, dict) or not models_cfg:
            raise ValueError("celltypist method config `models` must be a non-empty mapping")
        resolved = {str(level): _resolve_path(str(path), manifest_path=manifest_path) for level, path in models_cfg.items()}
    elif "model" in method_cfg:
        target = str(method_cfg.get("target_label_column") or "")
        if not target:
            if len(label_columns) != 1:
                raise ValueError(
                    "celltypist method config must set `target_label_column` when using a single `model` with multiple label columns"
                )
            target = label_columns[0]
        resolved = {target: _resolve_path(str(method_cfg["model"]), manifest_path=manifest_path)}
    else:
        raise ValueError("celltypist method config must provide either `model` or `models`")

    missing = [level for level in resolved if level not in label_columns]
    if missing:
        raise ValueError(f"celltypist models configured for unknown label columns: {', '.join(sorted(missing))}")
    return resolved


def run_celltypist(
    manifest: Dict[str, Any],
    *,
    output_dir: Path,
) -> Dict[str, Any]:
    import celltypist
    from celltypist import models

    del output_dir
    method_cfg = dict((manifest.get("method_configs") or {}).get("celltypist") or {})
    label_columns = list(manifest["label_columns"])
    manifest_path = Path(str(manifest["dataset_manifest_path"]))
    model_map = _resolve_model_map(method_cfg, label_columns=label_columns, manifest_path=manifest_path)
    majority_voting = bool(method_cfg.get("majority_voting", False))
    raw_mode = str(method_cfg.get("mode", "best_match"))
    mode_aliases = {
        "best_match": "best match",
        "best match": "best match",
        "prob_match": "prob match",
        "prob match": "prob match",
    }
    mode = mode_aliases.get(raw_mode)
    if mode is None:
        raise ValueError("celltypist method config `mode` must be `best_match`/`best match` or `prob_match`/`prob match`")

    query = read_h5ad(str(manifest["query_h5ad"]))
    true_df = query.obs.loc[:, list(model_map)].copy()

    load_start = time.perf_counter()
    loaded_models = {level: models.Model.load(model=path) for level, path in model_map.items()}
    load_elapsed = time.perf_counter() - load_start

    predict_start = time.perf_counter()
    pred_df = pd.DataFrame(index=query.obs_names)
    probability_summary: Dict[str, Dict[str, float]] = {}
    matched_features: Dict[str, int] = {}
    for level, model in loaded_models.items():
        annotation = celltypist.annotate(query, model=model, majority_voting=majority_voting, mode=mode)
        labels = annotation.predicted_labels.iloc[:, 0].astype(str).to_numpy()
        adata_pred = None
        if majority_voting:
            adata_pred = annotation.to_adata(insert_labels=True)
            if "majority_voting" in adata_pred.obs:
                labels = adata_pred.obs["majority_voting"].astype(str).to_numpy()
        probs = annotation.probability_matrix.to_numpy(dtype=np.float32)
        top1 = probs.max(axis=1).astype(np.float32)
        pred_df[f"pred_{level}"] = labels
        pred_df[f"conf_{level}"] = top1
        pred_df[f"margin_{level}"] = _top2_margin(probs)
        pred_df[f"is_unknown_{level}"] = False
        probability_summary[level] = {
            "mean_confidence": float(np.mean(top1)) if len(top1) else 0.0,
            "mean_margin": float(np.mean(pred_df[f"margin_{level}"])) if len(top1) else 0.0,
        }
        matched_features[level] = int(len(getattr(model.classifier, "features", [])))
    predict_elapsed = time.perf_counter() - predict_start

    metrics = evaluate_predictions(pred_df, true_df, list(model_map))
    behavior = evaluate_prediction_behavior(pred_df, true_df, list(model_map))
    metrics_by_domain = None
    behavior_by_domain = None
    domain_key = manifest.get("domain_key")
    if domain_key and domain_key in query.obs.columns:
        metrics_by_domain = evaluate_predictions_by_group(
            pred_df,
            true_df,
            list(model_map),
            group=query.obs[domain_key],
        )
        behavior_by_domain = evaluate_prediction_behavior_by_group(
            pred_df,
            true_df,
            list(model_map),
            group=query.obs[domain_key],
        )

    artifact_paths = {f"celltypist_model_{level}": path for level, path in model_map.items()}
    protocol_context = {
        "protocol_version": int(manifest.get("protocol_version", 1)),
        "random_seed": manifest.get("random_seed"),
        "split_name": manifest.get("split_name"),
        "split_description": manifest.get("split_description"),
        "reference_subset": manifest.get("reference_subset"),
        "query_subset": manifest.get("query_subset"),
        "domain_key": manifest.get("domain_key"),
    }
    return {
        "method": "celltypist",
        "dataset_name": manifest.get("dataset_name"),
        "dataset_version": manifest.get("version"),
        "protocol_version": int(manifest.get("protocol_version", 1)),
        "dataset_manifest": manifest.get("dataset_manifest_path"),
        "protocol_context": protocol_context,
        "label_columns": list(model_map),
        "metrics": metrics,
        "metrics_by_domain": metrics_by_domain,
        "behavior_metrics": behavior,
        "behavior_metrics_by_domain": behavior_by_domain,
        "hierarchy_metrics": None,
        "coordinate_metrics": None,
        "train_usage": _runtime_payload(phase="load_model", elapsed_seconds=load_elapsed, n_items=len(model_map)),
        "predict_usage": _runtime_payload(phase="predict", elapsed_seconds=predict_elapsed, n_items=query.n_obs),
        "artifact_sizes": None,
        "artifact_paths": artifact_paths,
        "artifact_checksums": artifact_checksums(artifact_paths),
        "model_source": "external_comparator",
        "model_input_path": artifact_paths,
        "train_config_used": method_cfg,
        "predict_config_used": {
            "majority_voting": majority_voting,
            "mode": mode,
            "target_label_columns": list(model_map),
        },
        "prediction_metadata": {
            "method_family": "published_comparator",
            "comparator_name": "celltypist",
            "majority_voting": majority_voting,
            "mode": mode,
            "matched_features": matched_features,
            "probability_summary": probability_summary,
        },
    }
