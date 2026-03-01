from __future__ import annotations

import json
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


def run_azimuth(
    manifest: Dict[str, Any],
    *,
    output_dir: Path,
) -> Dict[str, Any]:
    method_cfg = dict((manifest.get("method_configs") or {}).get("azimuth") or {})
    label_columns = list(manifest["label_columns"])
    target_label_column = str(method_cfg.get("target_label_column") or label_columns[-1])
    if target_label_column not in label_columns:
        raise ValueError(f"azimuth target_label_column not found in label_columns: {target_label_column}")

    manifest_path = Path(str(manifest["dataset_manifest_path"]))
    reference_h5ad = _resolve_path(str(manifest["reference_h5ad"]), manifest_path=manifest_path)
    query_h5ad = _resolve_path(str(manifest["query_h5ad"]), manifest_path=manifest_path)
    batch_key = str(method_cfg.get("batch_key") or manifest.get("domain_key") or "")
    reference_layer = str(method_cfg.get("reference_layer", "counts"))
    query_layer = str(method_cfg.get("query_layer", "counts"))
    nfeatures = int(method_cfg.get("nfeatures", 2000))
    npcs = int(method_cfg.get("npcs", 30))
    dims = list(method_cfg.get("dims", list(range(1, min(npcs, 30) + 1))))
    k_anchor = int(method_cfg.get("k_anchor", 5))
    k_score = int(method_cfg.get("k_score", 30))
    k_weight = int(method_cfg.get("k_weight", 50))
    n_trees = int(method_cfg.get("n_trees", 20))
    mapping_score_k = int(method_cfg.get("mapping_score_k", 100))
    reference_k_param = int(method_cfg.get("reference_k_param", max(5, min(30, k_weight))))
    sct_ncells = int(method_cfg.get("sct_ncells", 5000))
    umap_name = str(method_cfg.get("umap_name", "ref.umap"))
    homolog_table = method_cfg.get("homolog_table") or os.environ.get(
        "ATLASMTL_AZIMUTH_HOMOLOG_TABLE",
        "https://seurat.nygenome.org/azimuth/references/homologs.rds",
    )
    save_raw_outputs = bool(method_cfg.get("save_raw_outputs", False))

    azimuth_dir = output_dir / "azimuth"
    azimuth_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = azimuth_dir / "reference"
    predictions_csv = azimuth_dir / "predictions.csv"
    metadata_json = azimuth_dir / "metadata.json"
    config_json = azimuth_dir / "config.json"
    config_json.write_text(
        json.dumps(
            {
                "reference_h5ad": reference_h5ad,
                "query_h5ad": query_h5ad,
                "target_label_column": target_label_column,
                "batch_key": batch_key,
                "reference_layer": reference_layer,
                "query_layer": query_layer,
                "nfeatures": nfeatures,
                "npcs": npcs,
                "dims": dims,
                "k_anchor": k_anchor,
                "k_score": k_score,
                "k_weight": k_weight,
                "n_trees": n_trees,
                "mapping_score_k": mapping_score_k,
                "reference_k_param": reference_k_param,
                "sct_ncells": sct_ncells,
                "umap_name": umap_name,
                "homolog_table": homolog_table,
                "reference_dir": str(reference_dir),
                "output_predictions_csv": str(predictions_csv),
                "output_metadata_json": str(metadata_json),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.setdefault("ATLASMTL_PYTHON", "/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python")
    env.setdefault("ATLASMTL_AZIMUTH_LIB", "/home/data/fhz/seurat_v5")
    env.setdefault("R_LIBS_USER", "/home/data/fhz/seurat_v5")
    command = ["Rscript", "benchmark/methods/run_azimuth.R", "--config", str(config_json)]

    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
    )
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(
            "Azimuth-style comparator failed:\n"
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
        "azimuth_config": str(config_json),
        "azimuth_predictions": str(predictions_csv),
        "azimuth_metadata": str(metadata_json),
        "azimuth_reference_rds": str(reference_dir / "ref.Rds"),
        "azimuth_reference_annoy": str(reference_dir / "idx.annoy"),
    }
    if not save_raw_outputs:
        artifact_paths = {"azimuth_metadata": str(metadata_json)}

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
        "method": "azimuth",
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
        "train_usage": _runtime_payload(phase="transfer", elapsed_seconds=elapsed, n_items=query.n_obs),
        "predict_usage": _runtime_payload(phase="predict", elapsed_seconds=elapsed, n_items=query.n_obs),
        "artifact_sizes": None,
        "artifact_paths": artifact_paths,
        "artifact_checksums": artifact_checksums(artifact_paths),
        "model_source": "external_comparator",
        "model_input_path": {"reference_h5ad": reference_h5ad, "query_h5ad": query_h5ad},
        "train_config_used": method_cfg,
        "predict_config_used": {
            "target_label_column": target_label_column,
            "batch_key": batch_key,
            "reference_layer": reference_layer,
            "query_layer": query_layer,
            "nfeatures": nfeatures,
            "npcs": npcs,
            "k_anchor": k_anchor,
            "k_score": k_score,
            "k_weight": k_weight,
            "n_trees": n_trees,
            "mapping_score_k": mapping_score_k,
            "reference_k_param": reference_k_param,
            "sct_ncells": sct_ncells,
            "umap_name": umap_name,
            "homolog_table": homolog_table,
        },
        "prediction_metadata": {
            "method_family": "published_comparator",
            "comparator_name": "azimuth",
            "implementation_backend": metadata.get("implementation_backend", "azimuth_native"),
            "target_label_column": target_label_column,
            "batch_key": batch_key,
            "reference_layer": reference_layer,
            "query_layer": query_layer,
            "nfeatures": int(metadata.get("nfeatures", nfeatures)),
            "npcs": int(metadata.get("npcs", npcs)),
            "dims": metadata.get("dims", dims),
            "score_columns": metadata.get("score_columns", []),
            "umap_name": metadata.get("umap_name", umap_name),
            "reference_dir": metadata.get("reference_dir", str(reference_dir)),
        },
    }
