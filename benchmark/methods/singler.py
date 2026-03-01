from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

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


def run_singler(
    manifest: Dict[str, Any],
    *,
    output_dir: Path,
) -> Dict[str, Any]:
    method_cfg = dict((manifest.get("method_configs") or {}).get("singler") or {})
    label_columns = list(manifest["label_columns"])
    target_label_column = str(method_cfg.get("target_label_column") or label_columns[-1])
    if target_label_column not in label_columns:
        raise ValueError(f"singler target_label_column not found in label_columns: {target_label_column}")

    manifest_path = Path(str(manifest["dataset_manifest_path"]))
    reference_h5ad = _resolve_path(str(manifest["reference_h5ad"]), manifest_path=manifest_path)
    query_h5ad = _resolve_path(str(manifest["query_h5ad"]), manifest_path=manifest_path)
    reference_layer = str(method_cfg.get("reference_layer", "counts"))
    query_layer = str(method_cfg.get("query_layer", "counts"))
    normalize_log1p = bool(method_cfg.get("normalize_log1p", True))
    use_pruned_labels = bool(method_cfg.get("use_pruned_labels", True))
    fine_tune = bool(method_cfg.get("fine_tune", True))
    prune = bool(method_cfg.get("prune", True))
    quantile = float(method_cfg.get("quantile", 0.8))
    de_method = str(method_cfg.get("de_method", "classic"))
    save_raw_outputs = bool(method_cfg.get("save_raw_outputs", False))

    singler_dir = output_dir / "singler"
    singler_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = singler_dir / "predictions.csv"
    metadata_json = singler_dir / "metadata.json"
    config_json = singler_dir / "config.json"
    config_json.write_text(
        json.dumps(
            {
                "reference_h5ad": reference_h5ad,
                "query_h5ad": query_h5ad,
                "target_label_column": target_label_column,
                "reference_layer": reference_layer,
                "query_layer": query_layer,
                "normalize_log1p": normalize_log1p,
                "use_pruned_labels": use_pruned_labels,
                "fine_tune": fine_tune,
                "prune": prune,
                "quantile": quantile,
                "de_method": de_method,
                "output_predictions_csv": str(predictions_csv),
                "output_metadata_json": str(metadata_json),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    command = ["Rscript", "benchmark/methods/run_singler.R", "--config", str(config_json)]
    env = os.environ.copy()
    env.setdefault("ATLASMTL_PYTHON", "/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python")

    train_start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
    )
    train_elapsed = time.perf_counter() - train_start
    if completed.returncode != 0:
        raise RuntimeError(
            "SingleR comparator failed:\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    predictions = pd.read_csv(predictions_csv)
    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    query = read_h5ad(query_h5ad)
    true_df = query.obs.loc[:, [target_label_column]].copy()

    pred_df = pd.DataFrame(index=predictions["cell_id"].astype(str))
    pred_df[f"pred_{target_label_column}"] = predictions["predicted_label"].astype(str).to_numpy()
    pred_df[f"conf_{target_label_column}"] = predictions["conf"].fillna(0.0).astype(float).to_numpy()
    pred_df[f"margin_{target_label_column}"] = predictions["margin"].fillna(0.0).astype(float).to_numpy()
    pred_df[f"is_unknown_{target_label_column}"] = predictions["is_unknown"].astype(bool).to_numpy()
    pred_df = pred_df.loc[query.obs_names]

    metrics = evaluate_predictions(pred_df, true_df, [target_label_column])
    behavior = evaluate_prediction_behavior(pred_df, true_df, [target_label_column])
    metrics_by_domain = None
    behavior_by_domain = None
    domain_key = manifest.get("domain_key")
    if domain_key and domain_key in query.obs.columns:
        metrics_by_domain = evaluate_predictions_by_group(
            pred_df,
            true_df,
            [target_label_column],
            group=query.obs[domain_key],
        )
        behavior_by_domain = evaluate_prediction_behavior_by_group(
            pred_df,
            true_df,
            [target_label_column],
            group=query.obs[domain_key],
        )

    artifact_paths = {
        "singler_config": str(config_json),
        "singler_predictions": str(predictions_csv),
        "singler_metadata": str(metadata_json),
    }
    if not save_raw_outputs:
        artifact_paths = {"singler_metadata": str(metadata_json)}

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
        "method": "singler",
        "dataset_name": manifest.get("dataset_name"),
        "dataset_version": manifest.get("version"),
        "protocol_version": int(manifest.get("protocol_version", 1)),
        "dataset_manifest": manifest.get("dataset_manifest_path"),
        "protocol_context": protocol_context,
        "label_columns": [target_label_column],
        "metrics": metrics,
        "metrics_by_domain": metrics_by_domain,
        "behavior_metrics": behavior,
        "behavior_metrics_by_domain": behavior_by_domain,
        "hierarchy_metrics": None,
        "coordinate_metrics": None,
        "train_usage": _runtime_payload(phase="annotate", elapsed_seconds=train_elapsed, n_items=query.n_obs),
        "predict_usage": _runtime_payload(phase="predict", elapsed_seconds=train_elapsed, n_items=query.n_obs),
        "artifact_sizes": None,
        "artifact_paths": artifact_paths,
        "artifact_checksums": artifact_checksums(artifact_paths),
        "model_source": "external_comparator",
        "model_input_path": {"reference_h5ad": reference_h5ad, "query_h5ad": query_h5ad},
        "train_config_used": method_cfg,
        "predict_config_used": {
            "target_label_column": target_label_column,
            "reference_layer": reference_layer,
            "query_layer": query_layer,
            "use_pruned_labels": use_pruned_labels,
        },
        "prediction_metadata": {
            "method_family": "published_comparator",
            "comparator_name": "singler",
            "target_label_column": target_label_column,
            "reference_layer": reference_layer,
            "query_layer": query_layer,
            "normalize_log1p": normalize_log1p,
            "use_pruned_labels": use_pruned_labels,
            "fine_tune": fine_tune,
            "prune": prune,
            "quantile": quantile,
            "de_method": de_method,
            "unknown_count": int(metadata.get("unknown_count", 0)),
            "unknown_rate": float(metadata.get("unknown_rate", 0.0)),
            "n_shared_genes": int(metadata.get("n_shared_genes", 0)),
        },
    }
