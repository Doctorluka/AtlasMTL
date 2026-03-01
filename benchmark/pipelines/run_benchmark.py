#!/usr/bin/env python
"""Benchmark runner for atlasmtl and published comparator tools (incremental).

This runner is intentionally minimal: it focuses on a fair and reproducible
benchmark contract (metrics + artifacts + settings) before adding multiple
comparator wrappers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import yaml
from anndata import read_h5ad

from atlasmtl import TrainedModel, build_model, evaluate_predictions, predict
from atlasmtl.core.evaluate import (
    evaluate_coordinate_metrics,
    evaluate_hierarchy_metrics,
    evaluate_prediction_behavior,
    evaluate_prediction_behavior_by_group,
    evaluate_predictions_by_group,
)
from atlasmtl.models import artifact_checksums, resolve_manifest_paths
from atlasmtl.preprocess import PreprocessConfig, preprocess_query, preprocess_reference
from benchmark.methods import run_method

PROTOCOL_VERSION = 1
REQUIRED_MANIFEST_KEYS = {"reference_h5ad", "query_h5ad", "label_columns"}
OPTIONAL_MANIFEST_KEYS = {
    "dataset_name",
    "version",
    "protocol_version",
    "random_seed",
    "split_name",
    "split_description",
    "reference_subset",
    "query_subset",
    "coord_targets",
    "query_coord_targets",
    "domain_key",
    "train",
    "predict",
    "method_configs",
    "var_names_type",
    "species",
    "gene_id_table",
    "feature_space",
    "hvg_config",
    "duplicate_policy",
    "unmapped_policy",
}
TRAIN_CONFIG_KEYS = {
    "hidden_sizes",
    "dropout_rate",
    "batch_size",
    "num_epochs",
    "learning_rate",
    "input_transform",
    "val_fraction",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "random_state",
    "calibration_method",
    "calibration_max_iter",
    "calibration_lr",
    "reference_storage",
    "reference_path",
}
PREDICT_CONFIG_KEYS = {
    "knn_correction",
    "confidence_high",
    "confidence_low",
    "margin_threshold",
    "knn_k",
    "knn_conf_low",
    "knn_vote_mode",
    "knn_reference_mode",
    "knn_index_mode",
    "input_transform",
    "apply_calibration",
    "openset_method",
    "openset_threshold",
    "openset_label_column",
    "hierarchy_rules",
    "enforce_hierarchy",
    "batch_size",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--atlasmtl-model",
        help="Path to atlasmtl `model.pth` or `model_manifest.json` for benchmark runs.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["atlasmtl"],
        help="Benchmark methods to run. Comparator wrappers are added incrementally.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device used by atlasmtl runs. Default: auto.",
    )
    return parser.parse_args()


def _validate_allowed_keys(payload: Dict[str, Any], *, allowed: Iterable[str], label: str) -> None:
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise ValueError(f"{label} contains unsupported keys: {', '.join(unknown)}")


def _require_mapping(value: Any, *, key: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping when provided")
    return dict(value)


def _resolve_manifest_path(value: str, *, manifest_path: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((manifest_path.parent / path).resolve())


def _load_manifest(path: str) -> Dict[str, Any]:
    manifest_path = Path(path).resolve()
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dataset manifest must be a YAML mapping")
    _validate_allowed_keys(
        payload,
        allowed=REQUIRED_MANIFEST_KEYS | OPTIONAL_MANIFEST_KEYS,
        label="dataset manifest",
    )
    for key in REQUIRED_MANIFEST_KEYS:
        if key not in payload:
            raise ValueError(f"dataset manifest missing required key: {key}")
    if not isinstance(payload["label_columns"], list) or not payload["label_columns"]:
        raise ValueError("label_columns must be a non-empty list")
    if not all(isinstance(item, str) and item for item in payload["label_columns"]):
        raise ValueError("label_columns must contain non-empty strings")
    payload["train"] = _require_mapping(payload.get("train"), key="train")
    payload["predict"] = _require_mapping(payload.get("predict"), key="predict")
    payload["method_configs"] = _require_mapping(payload.get("method_configs"), key="method_configs")
    payload["coord_targets"] = _require_mapping(payload.get("coord_targets"), key="coord_targets")
    payload["query_coord_targets"] = _require_mapping(
        payload.get("query_coord_targets"),
        key="query_coord_targets",
    )
    _validate_allowed_keys(payload["train"], allowed=TRAIN_CONFIG_KEYS, label="train config")
    _validate_allowed_keys(payload["predict"], allowed=PREDICT_CONFIG_KEYS, label="predict config")
    if "protocol_version" in payload and int(payload["protocol_version"]) != PROTOCOL_VERSION:
        raise ValueError(f"protocol_version must be {PROTOCOL_VERSION}")
    payload["protocol_version"] = int(payload.get("protocol_version", PROTOCOL_VERSION))
    payload["reference_h5ad"] = _resolve_manifest_path(str(payload["reference_h5ad"]), manifest_path=manifest_path)
    payload["query_h5ad"] = _resolve_manifest_path(str(payload["query_h5ad"]), manifest_path=manifest_path)
    for key in ("reference_h5ad", "query_h5ad"):
        if not Path(str(payload[key])).exists():
            raise ValueError(f"{key} does not exist: {payload[key]}")
    payload["dataset_manifest_path"] = str(manifest_path)
    return payload


def _build_preprocess_config(manifest: Dict[str, Any]) -> Optional[PreprocessConfig]:
    var_names_type = manifest.get("var_names_type")
    species = manifest.get("species")
    if var_names_type is None and species is None:
        return None
    if var_names_type is None or species is None:
        raise ValueError("both var_names_type and species must be provided when preprocessing is configured")
    hvg_cfg = _require_mapping(manifest.get("hvg_config"), key="hvg_config")
    feature_space = str(manifest.get("feature_space", "hvg"))
    if feature_space not in {"hvg", "whole"}:
        raise ValueError("feature_space must be one of: hvg, whole")
    return PreprocessConfig(
        var_names_type=str(var_names_type),
        species=str(species),
        gene_id_table=manifest.get("gene_id_table"),
        feature_space=feature_space,
        n_top_genes=int(hvg_cfg.get("n_top_genes", 3000)),
        hvg_method=str(hvg_cfg.get("method", "seurat_v3")),
        hvg_batch_key=hvg_cfg.get("batch_key"),
        duplicate_policy=str(manifest.get("duplicate_policy", "sum")),
        unmapped_policy=str(manifest.get("unmapped_policy", "drop")),
    )


def _prepare_manifest_datasets(manifest: Dict[str, Any], *, output_dir: Path) -> Dict[str, Any]:
    preprocess_config = _build_preprocess_config(manifest)
    if preprocess_config is None:
        return manifest

    ref = read_h5ad(str(manifest["reference_h5ad"]))
    query = read_h5ad(str(manifest["query_h5ad"]))
    ref_pp, feature_panel, ref_report = preprocess_reference(ref, preprocess_config)
    query_pp, query_report = preprocess_query(query, feature_panel, preprocess_config)

    ref_path = output_dir / "reference_preprocessed.h5ad"
    query_path = output_dir / "query_preprocessed.h5ad"
    ref_pp.write_h5ad(ref_path)
    query_pp.write_h5ad(query_path)

    updated = dict(manifest)
    updated["reference_h5ad"] = str(ref_path.resolve())
    updated["query_h5ad"] = str(query_path.resolve())
    updated["preprocess"] = {
        "config": preprocess_config.to_dict(),
        "reference_report": ref_report.to_dict(),
        "query_report": query_report.to_dict(),
        "feature_panel": feature_panel.to_dict(),
    }
    return updated


def _artifact_sizes_mb(paths: Dict[str, Optional[str]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, p in paths.items():
        if not p:
            continue
        if not isinstance(p, (str, Path)):
            continue
        path = Path(p)
        if not path.exists():
            continue
        out[f"{key}_mb"] = float(path.stat().st_size) / (1024.0 * 1024.0)
    out["total_mb"] = float(sum(v for v in out.values() if isinstance(v, float)))
    return out


def _run_atlasmtl(
    manifest: Dict[str, Any],
    *,
    output_dir: Path,
    atlasmtl_model: Optional[str],
    device: str,
) -> Dict[str, Any]:
    ref = read_h5ad(str(manifest["reference_h5ad"]))
    query = read_h5ad(str(manifest["query_h5ad"]))
    label_columns = list(manifest["label_columns"])
    coord_targets = dict(manifest.get("coord_targets") or {})
    train_cfg = dict(manifest.get("train") or {})
    protocol_context = {
        "protocol_version": int(manifest.get("protocol_version", PROTOCOL_VERSION)),
        "random_seed": manifest.get("random_seed"),
        "split_name": manifest.get("split_name"),
        "split_description": manifest.get("split_description"),
        "reference_subset": manifest.get("reference_subset"),
        "query_subset": manifest.get("query_subset"),
        "domain_key": manifest.get("domain_key"),
    }

    if atlasmtl_model:
        model = TrainedModel.load(atlasmtl_model)
        model_input_path = str(Path(atlasmtl_model).resolve())
        if atlasmtl_model.endswith(".json"):
            artifact_paths = resolve_manifest_paths(atlasmtl_model)
        else:
            manifest_candidate = Path(str(atlasmtl_model).replace(".pth", "_manifest.json"))
            artifact_paths = resolve_manifest_paths(str(manifest_candidate)) if manifest_candidate.exists() else None
        model_source = "pretrained_artifact"
    else:
        model = build_model(
            adata=ref,
            label_columns=label_columns,
            coord_targets=coord_targets,
            hidden_sizes=train_cfg.get("hidden_sizes"),
            dropout_rate=float(train_cfg.get("dropout_rate", 0.3)),
            batch_size=int(train_cfg.get("batch_size", 256)),
            num_epochs=int(train_cfg.get("num_epochs", 40)),
            learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
            input_transform=str(train_cfg.get("input_transform", "binary")),
            val_fraction=float(train_cfg.get("val_fraction", 0.0)),
            early_stopping_patience=train_cfg.get("early_stopping_patience"),
            early_stopping_min_delta=float(train_cfg.get("early_stopping_min_delta", 0.0)),
            random_state=int(train_cfg.get("random_state", 42)),
            calibration_method=train_cfg.get("calibration_method"),
            calibration_max_iter=int(train_cfg.get("calibration_max_iter", 100)),
            calibration_lr=float(train_cfg.get("calibration_lr", 0.05)),
            reference_storage=str(train_cfg.get("reference_storage", "external")),
            reference_path=train_cfg.get("reference_path"),
            device=device,
            show_progress=False,
            show_summary=False,
        )
        model_out = output_dir / "atlasmtl_model.pth"
        model.save(str(model_out))
        artifact_paths = resolve_manifest_paths(str(output_dir / "atlasmtl_model_manifest.json"))
        model_input_path = str(model_out.resolve())
        model_source = "trained_in_runner"

    pred_cfg = dict(manifest.get("predict") or {})
    result = predict(
        model,
        query,
        knn_correction=str(pred_cfg.get("knn_correction", "low_conf_only")),
        confidence_high=float(pred_cfg.get("confidence_high", 0.7)),
        confidence_low=float(pred_cfg.get("confidence_low", 0.4)),
        margin_threshold=float(pred_cfg.get("margin_threshold", 0.2)),
        knn_k=int(pred_cfg.get("knn_k", 15)),
        knn_conf_low=float(pred_cfg.get("knn_conf_low", 0.6)),
        knn_vote_mode=str(pred_cfg.get("knn_vote_mode", "majority")),
        knn_reference_mode=str(pred_cfg.get("knn_reference_mode", "full")),
        knn_index_mode=str(pred_cfg.get("knn_index_mode", "exact")),
        input_transform=pred_cfg.get("input_transform"),
        apply_calibration=pred_cfg.get("apply_calibration"),
        openset_method=pred_cfg.get("openset_method"),
        openset_threshold=pred_cfg.get("openset_threshold"),
        openset_label_column=pred_cfg.get("openset_label_column"),
        hierarchy_rules=pred_cfg.get("hierarchy_rules"),
        enforce_hierarchy=bool(pred_cfg.get("enforce_hierarchy", False)),
        batch_size=int(pred_cfg.get("batch_size", 256)),
        device=device,
        show_progress=False,
        show_summary=False,
    )

    true_df = query.obs.loc[:, label_columns].copy()
    per_level = evaluate_predictions(result.predictions, true_df, label_columns)
    behavior = evaluate_prediction_behavior(result.predictions, true_df, label_columns)
    by_domain = None
    behavior_by_domain = None
    domain_key = manifest.get("domain_key")
    if domain_key and domain_key in query.obs.columns:
        by_domain = evaluate_predictions_by_group(
            result.predictions,
            true_df,
            label_columns,
            group=query.obs[domain_key],
        )
        behavior_by_domain = evaluate_prediction_behavior_by_group(
            result.predictions,
            true_df,
            label_columns,
            group=query.obs[domain_key],
        )

    train_usage = model.get_resource_usage()
    pred_usage = result.get_resource_usage()
    artifacts = _artifact_sizes_mb(artifact_paths) if artifact_paths else {}

    coord_metrics = {}
    query_coord_targets = dict(manifest.get("query_coord_targets") or {})
    coordinate_preds = {}
    coordinate_truth = {}
    for name, key in query_coord_targets.items():
        pred_key = f"X_pred_{name}"
        if pred_key in result.coordinates and key in query.obsm:
            coordinate_preds[name] = np.asarray(result.coordinates[pred_key], dtype=np.float32)
            coordinate_truth[name] = np.asarray(query.obsm[key], dtype=np.float32)
    if coordinate_preds:
        coord_metrics = evaluate_coordinate_metrics(coordinate_preds, coordinate_truth, n_neighbors=10)

    hierarchy_metrics = None
    hierarchy_rules = pred_cfg.get("hierarchy_rules")
    if hierarchy_rules:
        hierarchy_metrics = evaluate_hierarchy_metrics(
            result.predictions,
            true_df,
            label_columns,
            hierarchy_rules=hierarchy_rules,
        )

    payload: Dict[str, Any] = {
        "method": "atlasmtl",
        "dataset_name": manifest.get("dataset_name"),
        "dataset_version": manifest.get("version"),
        "protocol_version": int(manifest.get("protocol_version", PROTOCOL_VERSION)),
        "dataset_manifest": manifest.get("dataset_manifest_path"),
        "protocol_context": protocol_context,
        "preprocess": manifest.get("preprocess"),
        "label_columns": label_columns,
        "metrics": per_level,
        "metrics_by_domain": by_domain,
        "behavior_metrics": behavior,
        "behavior_metrics_by_domain": behavior_by_domain,
        "hierarchy_metrics": hierarchy_metrics,
        "coordinate_metrics": coord_metrics or None,
        "train_usage": train_usage,
        "predict_usage": pred_usage,
        "artifact_sizes": artifacts,
        "artifact_paths": artifact_paths,
        "artifact_checksums": artifact_checksums(artifact_paths) if artifact_paths else {},
        "model_source": model_source,
        "model_input_path": model_input_path,
        "train_config_used": train_cfg if not atlasmtl_model else None,
        "predict_config_used": pred_cfg,
        "prediction_metadata": result.metadata,
    }
    return payload


def _write_reports(payload: Dict[str, Any], output_dir: Path) -> None:
    (output_dir / "metrics.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    rows = []
    for result in payload.get("results", []):
        for level, metrics in (result.get("metrics") or {}).items():
            row = {
                "method": result.get("method"),
                "dataset_name": result.get("dataset_name"),
                "level": level,
                "split_name": (result.get("protocol_context") or {}).get("split_name"),
                "model_source": result.get("model_source"),
                "knn_vote_mode": (result.get("predict_config_used") or {}).get("knn_vote_mode", "majority"),
                "knn_reference_mode": (result.get("predict_config_used") or {}).get("knn_reference_mode", "full"),
                "knn_index_mode": (result.get("predict_config_used") or {}).get("knn_index_mode", "exact"),
                "enforce_hierarchy": bool((result.get("predict_config_used") or {}).get("enforce_hierarchy", False)),
                "openset_method": (result.get("predict_config_used") or {}).get("openset_method"),
            }
            row.update(metrics)
            row.update((result.get("behavior_metrics") or {}).get(level, {}))
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "summary.csv", index=False)

    domain_rows = []
    for result in payload.get("results", []):
        by_domain = result.get("metrics_by_domain") or {}
        behavior_by_domain = result.get("behavior_metrics_by_domain") or {}
        for domain, per_level in by_domain.items():
            for level, metrics in (per_level or {}).items():
                row = {
                    "method": result.get("method"),
                    "dataset_name": result.get("dataset_name"),
                    "domain": domain,
                    "level": level,
                    "split_name": (result.get("protocol_context") or {}).get("split_name"),
                    "knn_vote_mode": (result.get("predict_config_used") or {}).get("knn_vote_mode", "majority"),
                    "knn_reference_mode": (result.get("predict_config_used") or {}).get("knn_reference_mode", "full"),
                    "knn_index_mode": (result.get("predict_config_used") or {}).get("knn_index_mode", "exact"),
                    "enforce_hierarchy": bool((result.get("predict_config_used") or {}).get("enforce_hierarchy", False)),
                }
                row.update(metrics)
                row.update((behavior_by_domain.get(domain) or {}).get(level, {}))
                domain_rows.append(row)
    if domain_rows:
        pd.DataFrame(domain_rows).to_csv(output_dir / "summary_by_domain.csv", index=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _prepare_manifest_datasets(_load_manifest(args.dataset_manifest), output_dir=output_dir)
    methods = list(args.methods)

    results: Dict[str, Any] = {
        "protocol_version": int(manifest.get("protocol_version", PROTOCOL_VERSION)),
        "dataset_manifest": str(Path(args.dataset_manifest).resolve()),
        "dataset_name": manifest.get("dataset_name"),
        "dataset_version": manifest.get("version"),
        "protocol_context": {
            "random_seed": manifest.get("random_seed"),
            "split_name": manifest.get("split_name"),
            "split_description": manifest.get("split_description"),
            "reference_subset": manifest.get("reference_subset"),
            "query_subset": manifest.get("query_subset"),
            "domain_key": manifest.get("domain_key"),
        },
        "preprocess": manifest.get("preprocess"),
        "results": [],
    }
    for method in methods:
        if method == "atlasmtl":
            payload = _run_atlasmtl(
                manifest,
                output_dir=output_dir,
                atlasmtl_model=args.atlasmtl_model,
                device=args.device,
            )
        else:
            payload = run_method(
                method,
                manifest,
                output_dir=output_dir,
                atlasmtl_model=args.atlasmtl_model,
                device=args.device,
            )
        results["results"].append(payload)

    _write_reports(results, output_dir)
    run_manifest = {
        "schema_version": 1,
        "protocol_version": results["protocol_version"],
        "dataset_manifest": results["dataset_manifest"],
        "dataset_name": manifest.get("dataset_name"),
        "dataset_version": manifest.get("version"),
        "random_seed": manifest.get("random_seed"),
        "split_name": manifest.get("split_name"),
        "split_description": manifest.get("split_description"),
        "reference_subset": manifest.get("reference_subset"),
        "query_subset": manifest.get("query_subset"),
        "domain_key": manifest.get("domain_key"),
        "methods": methods,
        "device": args.device,
        "atlasmtl_model": None if args.atlasmtl_model is None else str(Path(args.atlasmtl_model).resolve()),
        "python": str(Path(sys.executable).resolve()),
        "preprocess": manifest.get("preprocess"),
    }
    # Best-effort checksums for any artifacts written by the runner.
    artifact_paths = resolve_manifest_paths(str(output_dir / "atlasmtl_model_manifest.json")) if (output_dir / "atlasmtl_model_manifest.json").exists() else {}
    if artifact_paths:
        run_manifest["artifact_paths"] = artifact_paths
        run_manifest["artifact_checksums"] = artifact_checksums(artifact_paths)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(_json_safe(run_manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
