from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from anndata import read_h5ad
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from atlasmtl.core.data import extract_matrix
from atlasmtl.core.evaluate import evaluate_predictions


def _runtime_payload(*, phase: str, elapsed_seconds: float, n_items: int) -> Dict[str, object]:
    throughput = float(n_items / elapsed_seconds) if elapsed_seconds > 0 else None
    return {
        "phase": phase,
        "elapsed_seconds": float(elapsed_seconds),
        "items_per_second": throughput,
        "process_peak_rss_gb": None,
        "gpu_peak_memory_gb": None,
    }


def _prediction_frame(
    classifiers: Dict[str, KNeighborsClassifier],
    encoders: Dict[str, LabelEncoder],
    X_query: np.ndarray,
    index: pd.Index,
) -> pd.DataFrame:
    pred_df = pd.DataFrame(index=index)
    for col, clf in classifiers.items():
        probs = clf.predict_proba(X_query)
        pred_idx = probs.argmax(axis=1)
        classes = encoders[col].inverse_transform(pred_idx)
        sorted_probs = np.sort(probs, axis=1)
        top1 = sorted_probs[:, -1]
        top2 = sorted_probs[:, -2] if sorted_probs.shape[1] > 1 else np.zeros_like(top1)
        pred_df[f"pred_{col}"] = classes
        pred_df[f"conf_{col}"] = top1.astype(np.float32)
        pred_df[f"margin_{col}"] = (top1 - top2).astype(np.float32)
        pred_df[f"is_unknown_{col}"] = False
    return pred_df


def run_reference_knn(
    manifest: Dict[str, Any],
    *,
    output_dir: Path,
) -> Dict[str, Any]:
    method_cfg = dict((manifest.get("method_configs") or {}).get("reference_knn") or {})
    input_transform = str(method_cfg.get("input_transform", "binary"))
    k = int(method_cfg.get("k", 15))

    ref = read_h5ad(str(manifest["reference_h5ad"]))
    query = read_h5ad(str(manifest["query_h5ad"]))
    label_columns = list(manifest["label_columns"])

    train_start = time.perf_counter()
    X_ref = extract_matrix(ref, input_transform=input_transform)
    encoders: Dict[str, LabelEncoder] = {}
    classifiers: Dict[str, KNeighborsClassifier] = {}
    for col in label_columns:
        encoder = LabelEncoder()
        y_ref = encoder.fit_transform(ref.obs[col].astype(str).to_numpy())
        clf = KNeighborsClassifier(n_neighbors=min(k, len(X_ref)), weights="distance")
        clf.fit(X_ref, y_ref)
        encoders[col] = encoder
        classifiers[col] = clf
    train_elapsed = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    X_query = extract_matrix(query, train_genes=list(ref.var_names), input_transform=input_transform)
    pred_df = _prediction_frame(classifiers, encoders, X_query, query.obs_names)
    predict_elapsed = time.perf_counter() - predict_start

    true_df = query.obs.loc[:, label_columns].copy()
    metrics = evaluate_predictions(pred_df, true_df, label_columns)
    return {
        "method": "reference_knn",
        "dataset_name": manifest.get("dataset_name"),
        "dataset_version": manifest.get("version"),
        "protocol_version": int(manifest.get("protocol_version", 1)),
        "dataset_manifest": manifest.get("dataset_manifest_path"),
        "protocol_context": {
            "protocol_version": int(manifest.get("protocol_version", 1)),
            "random_seed": manifest.get("random_seed"),
            "split_name": manifest.get("split_name"),
            "split_description": manifest.get("split_description"),
            "reference_subset": manifest.get("reference_subset"),
            "query_subset": manifest.get("query_subset"),
            "domain_key": manifest.get("domain_key"),
        },
        "label_columns": label_columns,
        "metrics": metrics,
        "metrics_by_domain": None,
        "behavior_metrics": None,
        "behavior_metrics_by_domain": None,
        "hierarchy_metrics": None,
        "coordinate_metrics": None,
        "train_usage": _runtime_payload(phase="train", elapsed_seconds=train_elapsed, n_items=ref.n_obs),
        "predict_usage": _runtime_payload(phase="predict", elapsed_seconds=predict_elapsed, n_items=query.n_obs),
        "artifact_sizes": None,
        "artifact_paths": None,
        "artifact_checksums": {},
        "model_source": "reference_knn_baseline",
        "model_input_path": None,
        "train_config_used": method_cfg,
        "predict_config_used": {"input_transform": input_transform, "k": k},
        "prediction_metadata": {
            "input_transform": input_transform,
            "k": int(k),
            "method_family": "local_baseline",
        },
    }
