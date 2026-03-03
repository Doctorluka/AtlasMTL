from __future__ import annotations

import json
import os
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
from atlasmtl.utils.monitoring import run_subprocess_monitored
from benchmark.methods.config import resolve_counts_layer, resolve_reference_query_layers
from benchmark.methods.result_schema import build_input_contract


def _resolve_path(value: str, *, manifest_path: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((manifest_path.parent / path).resolve())


def run_seurat_anchor_transfer(
    manifest: Dict[str, Any],
    *,
    output_dir: Path,
) -> Dict[str, Any]:
    method_cfg = dict(
        (manifest.get("method_configs") or {}).get("seurat_anchor_transfer")
        or (manifest.get("method_configs") or {}).get("azimuth")
        or {}
    )
    counts_layer = resolve_counts_layer(manifest, method_cfg)
    label_columns = list(manifest["label_columns"])
    target_label_column = str(method_cfg.get("target_label_column") or label_columns[-1])
    if target_label_column not in label_columns:
        raise ValueError(
            "seurat_anchor_transfer target_label_column not found in label_columns: "
            f"{target_label_column}"
        )

    manifest_path = Path(str(manifest["dataset_manifest_path"]))
    reference_h5ad = _resolve_path(str(manifest["reference_h5ad"]), manifest_path=manifest_path)
    query_h5ad = _resolve_path(str(manifest["query_h5ad"]), manifest_path=manifest_path)
    batch_key = str(method_cfg.get("batch_key") or manifest.get("domain_key") or "")
    reference_layer, query_layer = resolve_reference_query_layers(manifest, method_cfg)
    nfeatures = int(method_cfg.get("nfeatures", 3000))
    npcs = int(method_cfg.get("npcs", 30))
    dims = list(method_cfg.get("dims", list(range(1, min(npcs, 30) + 1))))
    k_anchor = int(method_cfg.get("k_anchor", 5))
    k_score = int(method_cfg.get("k_score", 30))
    k_weight = int(method_cfg.get("k_weight", 50))
    integration_reduction = str(method_cfg.get("integration_reduction", "rpca"))
    normalization_method = str(method_cfg.get("normalization_method", "LogNormalize"))
    save_raw_outputs = bool(method_cfg.get("save_raw_outputs", False))

    seurat_dir = output_dir / "seurat_anchor_transfer"
    seurat_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = seurat_dir / "predictions.csv"
    metadata_json = seurat_dir / "metadata.json"
    reference_rds = seurat_dir / "reference.rds"
    mapped_query_rds = seurat_dir / "mapped_query.rds"
    config_json = seurat_dir / "config.json"
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
                "integration_reduction": integration_reduction,
                "normalization_method": normalization_method,
                "output_predictions_csv": str(predictions_csv),
                "output_metadata_json": str(metadata_json),
                "output_reference_rds": str(reference_rds),
                "output_mapped_query_rds": str(mapped_query_rds),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.setdefault("ATLASMTL_PYTHON", "/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python")
    env.setdefault("ATLASMTL_SEURAT_LIB", "/home/data/fhz/seurat_v5")
    env.setdefault("R_LIBS_USER", "/home/data/fhz/seurat_v5")
    command = ["Rscript", "benchmark/methods/run_seurat_anchor_transfer.R", "--config", str(config_json)]

    completed, runtime_usage = run_subprocess_monitored(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        phase="transfer",
        n_items=int(read_h5ad(query_h5ad).n_obs),
        device="cpu",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Seurat anchor transfer comparator failed:\n"
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
        "seurat_anchor_transfer_config": str(config_json),
        "seurat_anchor_transfer_predictions": str(predictions_csv),
        "seurat_anchor_transfer_metadata": str(metadata_json),
        "seurat_anchor_transfer_reference_rds": str(reference_rds),
        "seurat_anchor_transfer_mapped_query_rds": str(mapped_query_rds),
    }
    if not save_raw_outputs:
        artifact_paths = {"seurat_anchor_transfer_metadata": str(metadata_json)}

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
        "method": "seurat_anchor_transfer",
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
        "train_usage": runtime_usage,
        "predict_usage": {**runtime_usage, "phase": "predict"},
        "artifact_sizes": None,
        "artifact_paths": artifact_paths,
        "artifact_checksums": artifact_checksums(artifact_paths),
        "model_source": "external_comparator",
        "model_input_path": {"reference_h5ad": reference_h5ad, "query_h5ad": query_h5ad},
        "input_contract": build_input_contract(
            reference_matrix_source=f"layers/{reference_layer}",
            query_matrix_source=f"layers/{query_layer}",
            counts_layer=counts_layer,
            feature_alignment="prepared_shared_gene_panel",
            normalization_mode="seurat_anchor_transfer_internal_lognormalize",
            label_scope="single_level",
            backend=metadata.get("implementation_backend", "seurat_anchor_transfer"),
        ),
        "train_config_used": method_cfg,
        "predict_config_used": {
            "target_label_column": target_label_column,
            "batch_key": batch_key,
            "counts_layer": counts_layer,
            "reference_layer": reference_layer,
            "query_layer": query_layer,
            "matrix_source": f"layers/{query_layer}",
            "nfeatures": nfeatures,
            "npcs": npcs,
            "dims": dims,
            "k_anchor": k_anchor,
            "k_score": k_score,
            "k_weight": k_weight,
            "integration_reduction": integration_reduction,
            "normalization_method": normalization_method,
        },
        "prediction_metadata": {
            "method_family": "published_comparator",
            "comparator_name": "seurat_anchor_transfer",
            "implementation_backend": metadata.get("implementation_backend", "seurat_anchor_transfer"),
            "target_label_column": target_label_column,
            "batch_key": batch_key,
            "counts_layer": counts_layer,
            "reference_layer": reference_layer,
            "query_layer": query_layer,
            "matrix_source": f"layers/{query_layer}",
            "nfeatures": int(metadata.get("nfeatures", nfeatures)),
            "npcs": int(metadata.get("npcs", npcs)),
            "dims": metadata.get("dims", dims),
            "k_anchor": int(metadata.get("k_anchor", k_anchor)),
            "k_score": int(metadata.get("k_score", k_score)),
            "k_weight": int(metadata.get("k_weight", k_weight)),
            "integration_reduction": metadata.get("integration_reduction", integration_reduction),
            "reference_build_mode": metadata.get("reference_build_mode", "single_reference_pca"),
            "score_columns": metadata.get("score_columns", []),
        },
    }
