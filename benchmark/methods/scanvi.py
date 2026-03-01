from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData, read_h5ad

from atlasmtl.core.evaluate import (
    evaluate_prediction_behavior,
    evaluate_prediction_behavior_by_group,
    evaluate_predictions,
    evaluate_predictions_by_group,
)
from atlasmtl.models.checksums import sha256_file


def _runtime_payload(*, phase: str, elapsed_seconds: float, n_items: int) -> Dict[str, object]:
    throughput = float(n_items / elapsed_seconds) if elapsed_seconds > 0 else None
    return {
        "phase": phase,
        "elapsed_seconds": float(elapsed_seconds),
        "items_per_second": throughput,
        "process_peak_rss_gb": None,
        "gpu_peak_memory_gb": None,
    }


def _resolve_device(device: str) -> Tuple[str, int]:
    if device == "cuda":
        return "gpu", 1
    return "cpu", 1


def _ensure_batch_column(ref: AnnData, query: AnnData, batch_key: str | None) -> str:
    if batch_key and batch_key in ref.obs.columns and batch_key in query.obs.columns:
        return batch_key
    fallback = "_benchmark_batch"
    ref.obs[fallback] = "reference"
    query.obs[fallback] = "query"
    return fallback


def _prepare_reference(
    ref: AnnData,
    *,
    label_column: str,
    batch_key: str,
    unlabeled_category: str,
) -> AnnData:
    adata = ref.copy()
    labels = adata.obs[label_column].astype(str).fillna(unlabeled_category)
    adata.obs["_scanvi_label"] = labels
    return adata


def _prepare_query(query: AnnData, *, batch_key: str) -> AnnData:
    adata = query.copy()
    if batch_key not in adata.obs.columns:
        adata.obs[batch_key] = "query"
    return adata


def run_scanvi(
    manifest: Dict[str, Any],
    *,
    output_dir: Path,
    device: str,
) -> Dict[str, Any]:
    import scvi

    method_cfg = dict((manifest.get("method_configs") or {}).get("scanvi") or {})
    label_columns = list(manifest["label_columns"])
    label_column = str(method_cfg.get("target_label_column") or label_columns[-1])
    if label_column not in label_columns:
        raise ValueError(f"scanvi target_label_column not found in label_columns: {label_column}")

    ref = read_h5ad(str(manifest["reference_h5ad"]))
    query = read_h5ad(str(manifest["query_h5ad"]))

    batch_key = _ensure_batch_column(ref, query, str(method_cfg.get("batch_key") or manifest.get("domain_key") or ""))
    unlabeled_category = str(method_cfg.get("unlabeled_category", "Unknown"))
    n_latent = int(method_cfg.get("n_latent", 10))
    batch_size = int(method_cfg.get("batch_size", 128))
    train_size = float(method_cfg.get("train_size", 0.9))
    validation_size = float(method_cfg.get("validation_size", 0.1))
    scvi_epochs = int(method_cfg.get("scvi_max_epochs", 20))
    scanvi_epochs = int(method_cfg.get("scanvi_max_epochs", 20))
    query_epochs = int(method_cfg.get("query_max_epochs", 20))
    n_samples_per_label = method_cfg.get("n_samples_per_label")
    save_model = bool(method_cfg.get("save_model", False))

    accelerator, devices = _resolve_device(device)
    ref_adata = _prepare_reference(ref, label_column=label_column, batch_key=batch_key, unlabeled_category=unlabeled_category)
    query_adata = _prepare_query(query, batch_key=batch_key)

    train_start = time.perf_counter()
    scvi.model.SCVI.setup_anndata(ref_adata, batch_key=batch_key, labels_key="_scanvi_label")
    scvi_model = scvi.model.SCVI(ref_adata, n_latent=n_latent)
    scvi_model.train(
        max_epochs=scvi_epochs,
        accelerator=accelerator,
        devices=devices,
        train_size=train_size,
        validation_size=validation_size,
        batch_size=batch_size,
        early_stopping=False,
        enable_progress_bar=False,
    )
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        labels_key="_scanvi_label",
        unlabeled_category=unlabeled_category,
        adata=ref_adata,
    )
    scanvi_model.train(
        max_epochs=scanvi_epochs,
        n_samples_per_label=n_samples_per_label,
        accelerator=accelerator,
        devices=devices,
        train_size=train_size,
        validation_size=validation_size,
        batch_size=batch_size,
        enable_progress_bar=False,
    )
    train_elapsed = time.perf_counter() - train_start

    artifact_paths = None
    artifact_checksums = {}
    if save_model:
        model_dir = output_dir / "scanvi_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        scanvi_model.save(str(model_dir), overwrite=True)
        metadata_path = output_dir / "scanvi_model_metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "label_column": label_column,
                    "batch_key": batch_key,
                    "n_latent": n_latent,
                    "scvi_max_epochs": scvi_epochs,
                    "scanvi_max_epochs": scanvi_epochs,
                    "query_max_epochs": query_epochs,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        artifact_paths = {"scanvi_model_dir": str(model_dir), "scanvi_metadata": str(metadata_path)}
        artifact_checksums = {"scanvi_metadata": sha256_file(str(metadata_path))}

    predict_start = time.perf_counter()
    scvi.model.SCANVI.prepare_query_anndata(query_adata, scanvi_model)
    query_model = scvi.model.SCANVI.load_query_data(
        query_adata,
        scanvi_model,
        accelerator=accelerator,
        device=devices,
    )
    if query_epochs > 0:
        query_train_size = min(train_size, 0.75) if query_adata.n_obs < 10 else train_size
        query_validation_size = max(1.0 - query_train_size, 0.0)
        if query_validation_size <= 0:
            query_train_size = 0.75
            query_validation_size = 0.25
        query_model.train(
            max_epochs=query_epochs,
            accelerator=accelerator,
            devices=devices,
            train_size=query_train_size,
            validation_size=query_validation_size,
            batch_size=min(batch_size, max(query_adata.n_obs, 1)),
            enable_progress_bar=False,
            plan_kwargs={"weight_decay": 0.0},
        )
    pred_labels = np.asarray(query_model.predict(), dtype=object)
    soft_pred = query_model.predict(soft=True)
    if isinstance(soft_pred, tuple):
        soft_pred = soft_pred[0]
    if isinstance(soft_pred, pd.DataFrame):
        prob_df = soft_pred
    else:
        classes = sorted(pd.unique(ref_adata.obs["_scanvi_label"].astype(str)))
        prob_df = pd.DataFrame(np.asarray(soft_pred), index=query_adata.obs_names, columns=classes)
    probs = prob_df.to_numpy(dtype=np.float32)
    top1 = probs.max(axis=1).astype(np.float32)
    sorted_probs = np.sort(probs, axis=1)
    top2 = sorted_probs[:, -2] if sorted_probs.shape[1] > 1 else np.zeros_like(top1)
    latent = query_model.get_latent_representation()
    predict_elapsed = time.perf_counter() - predict_start

    pred_df = pd.DataFrame(index=query_adata.obs_names)
    pred_df[f"pred_{label_column}"] = pred_labels.astype(str)
    pred_df[f"conf_{label_column}"] = top1
    pred_df[f"margin_{label_column}"] = (top1 - top2).astype(np.float32)
    pred_df[f"is_unknown_{label_column}"] = False

    true_df = query.obs.loc[:, [label_column]].copy()
    metrics = evaluate_predictions(pred_df, true_df, [label_column])
    behavior = evaluate_prediction_behavior(pred_df, true_df, [label_column])
    metrics_by_domain = None
    behavior_by_domain = None
    domain_key = manifest.get("domain_key")
    if domain_key and domain_key in query.obs.columns:
        metrics_by_domain = evaluate_predictions_by_group(
            pred_df,
            true_df,
            [label_column],
            group=query.obs[domain_key],
        )
        behavior_by_domain = evaluate_prediction_behavior_by_group(
            pred_df,
            true_df,
            [label_column],
            group=query.obs[domain_key],
        )

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
        "method": "scanvi",
        "dataset_name": manifest.get("dataset_name"),
        "dataset_version": manifest.get("version"),
        "protocol_version": int(manifest.get("protocol_version", 1)),
        "dataset_manifest": manifest.get("dataset_manifest_path"),
        "protocol_context": protocol_context,
        "label_columns": [label_column],
        "metrics": metrics,
        "metrics_by_domain": metrics_by_domain,
        "behavior_metrics": behavior,
        "behavior_metrics_by_domain": behavior_by_domain,
        "hierarchy_metrics": None,
        "coordinate_metrics": None,
        "train_usage": _runtime_payload(phase="train", elapsed_seconds=train_elapsed, n_items=ref_adata.n_obs),
        "predict_usage": _runtime_payload(phase="predict", elapsed_seconds=predict_elapsed, n_items=query_adata.n_obs),
        "artifact_sizes": None,
        "artifact_paths": artifact_paths,
        "artifact_checksums": artifact_checksums,
        "model_source": "trained_in_runner",
        "model_input_path": None,
        "train_config_used": method_cfg,
        "predict_config_used": {
            "target_label_column": label_column,
            "batch_key": batch_key,
            "query_max_epochs": query_epochs,
        },
        "prediction_metadata": {
            "method_family": "published_comparator",
            "comparator_name": "scanvi",
            "label_column": label_column,
            "batch_key": batch_key,
            "n_latent": n_latent,
            "latent_shape": list(latent.shape),
            "probability_classes": list(prob_df.columns.astype(str)),
        },
    }
