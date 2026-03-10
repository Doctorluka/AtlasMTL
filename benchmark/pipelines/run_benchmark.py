#!/usr/bin/env python
"""Benchmark runner for atlasmtl and published comparator tools (incremental).

This runner is intentionally minimal: it focuses on a fair and reproducible
benchmark contract (metrics + artifacts + settings) before adding multiple
comparator wrappers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import yaml
from anndata import read_h5ad

from atlasmtl import TrainedModel, build_model, evaluate_predictions, predict
from atlasmtl.mapping import (
    build_parent_conditioned_refinement_plan,
    discover_hotspot_parents,
    fit_parent_conditioned_reranker,
    get_refinement_guardrail_profile,
    suggest_parent_conditioned_refinement,
    suggest_task_weight_schedule,
)
from atlasmtl.core.evaluate import (
    evaluate_coordinate_metrics,
    evaluate_hierarchy_metrics,
    evaluate_prediction_behavior,
    evaluate_prediction_behavior_by_group,
    evaluate_predictions_by_group,
)
from atlasmtl.models import artifact_checksums, resolve_manifest_paths
from atlasmtl.preprocess import PreprocessConfig, preprocess_query, preprocess_reference
from benchmark.methods.atlasmtl_inputs import (
    adata_with_matrix_from_layer,
    matrix_source_label,
    resolve_atlasmtl_layer_config,
    resolve_task_weight_candidates,
    resolve_task_weights,
    select_task_weight_candidate_from_summary,
    task_weight_scheme_name,
)
from benchmark.methods.result_schema import build_input_contract
from benchmark.methods import run_method

PROTOCOL_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
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
    "input_matrix_type",
    "counts_layer",
    "gene_id_table",
    "canonical_target",
    "ensembl_source_column",
    "symbol_source_column",
    "prefer_existing_ensembl_column",
    "mapping_table_kind",
    "report_unmapped_top_n",
    "counts_check_tiny_positive_tol",
    "counts_confirm_fraction",
    "feature_space",
    "hvg_config",
    "duplicate_policy",
    "unmapped_policy",
    "experiment_round",
    "optimization_stage",
    "config_name",
    "seed",
    "source_formal_manifest",
    "generated_from_manifest_index",
}
TRAIN_CONFIG_KEYS = {
    "hidden_sizes",
    "dropout_rate",
    "batch_size",
    "num_epochs",
    "learning_rate",
    "optimizer_name",
    "weight_decay",
    "scheduler_name",
    "scheduler_factor",
    "scheduler_patience",
    "scheduler_min_lr",
    "scheduler_monitor",
    "input_transform",
    "val_fraction",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "random_state",
    "calibration_method",
    "calibration_max_iter",
    "calibration_lr",
    "class_weighting",
    "class_balanced_sampling",
    "parent_conditioned_child_correction",
    "reference_storage",
    "reference_path",
    "init_model_path",
    "task_weights",
    "task_weight_policy",
    "task_weight_selector",
    "task_weight_policy_source_run",
    "task_weight_candidates",
    "knn_reference_obsm_key",
    "knn_space",
}
PREDICT_CONFIG_KEYS = {
    "knn_correction",
    "knn_query_obsm_key",
    "knn_space",
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
    "refinement_policy",
    "refinement_plan_path",
    "refinement_artifact_path",
    "refinement_parent_level",
    "refinement_child_level",
    "hotspot_selection_mode",
    "hotspot_top_k",
    "hotspot_cumulative_target",
    "hotspot_min_cells_per_parent",
    "hotspot_max_selected_parents",
    "refinement_guardrail_profile",
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


def _resolve_code_version() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def _effective_variant_name(result: Dict[str, Any]) -> Any:
    ablation = result.get("ablation_config") or {}
    return ablation.get("variant_name") or result.get("variant_name")


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
    path = path.expanduser()
    manifest_relative = (manifest_path.parent / path).resolve()
    if manifest_relative.exists():
        return str(manifest_relative)
    repo_relative = (REPO_ROOT / path).resolve()
    return str(repo_relative)


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
        canonical_target=str(manifest.get("canonical_target", "ensembl")),
        ensembl_source_column=manifest.get("ensembl_source_column"),
        symbol_source_column=manifest.get("symbol_source_column"),
        prefer_existing_ensembl_column=bool(manifest.get("prefer_existing_ensembl_column", True)),
        mapping_table_kind=str(manifest.get("mapping_table_kind", "biomart_human_mouse_rat")),
        input_matrix_type=str(manifest.get("input_matrix_type", "infer")),
        counts_layer=str(manifest.get("counts_layer", "counts")),
        gene_id_table=manifest.get("gene_id_table"),
        report_unmapped_top_n=int(manifest.get("report_unmapped_top_n", 20)),
        counts_check_tiny_positive_tol=float(manifest.get("counts_check_tiny_positive_tol", 1e-8)),
        counts_confirm_fraction=float(manifest.get("counts_confirm_fraction", 0.999)),
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


def _normalize_knn_correction(value: Any) -> str:
    if isinstance(value, bool):
        return "all" if value else "off"
    if value is None:
        return "off"
    return str(value)


def _infer_parent_child_levels(
    label_columns: list[str],
    pred_cfg: Dict[str, Any],
) -> tuple[Optional[str], Optional[str]]:
    parent_level = pred_cfg.get("refinement_parent_level")
    child_level = pred_cfg.get("refinement_child_level")
    if parent_level and child_level:
        return str(parent_level), str(child_level)
    if len(label_columns) < 2:
        return None, None
    return str(label_columns[-2]), str(label_columns[-1])


def _edge_breakdown(
    pred_df: pd.DataFrame,
    true_df: pd.DataFrame,
    *,
    parent_col: str,
    child_col: str,
    child_to_parent: Dict[str, str],
) -> Dict[str, float]:
    unknown = "Unknown"
    parent_true = true_df[parent_col].astype(str)
    child_true = true_df[child_col].astype(str)
    parent_pred = pred_df[f"pred_{parent_col}"].astype(str)
    child_pred = pred_df[f"pred_{child_col}"].astype(str)
    implied_parent = child_pred.map(lambda x: child_to_parent.get(x) if x != unknown else unknown).fillna("MISSING_PARENT")
    parent_correct = parent_true == parent_pred
    child_correct = child_true == child_pred
    path_break = (child_pred != unknown) & (parent_pred != unknown) & (implied_parent != parent_pred)
    return {
        "parent_accuracy": float((parent_true == parent_pred).mean()),
        "child_accuracy": float((child_true == child_pred).mean()),
        "parent_correct_child_wrong_rate": float((parent_correct & ~child_correct).mean()),
        "parent_wrong_child_correct_rate": float((~parent_correct & child_correct).mean()),
        "both_wrong_rate": float((~parent_correct & ~child_correct).mean()),
        "child_unknown_rate": float((child_pred == unknown).mean()),
        "parent_unknown_rate": float((parent_pred == unknown).mean()),
        "path_break_rate": float(path_break.mean()),
    }


def _subtree_breakdown(
    pred_df: pd.DataFrame,
    true_df: pd.DataFrame,
    *,
    parent_col: str,
    child_col: str,
    child_to_parent: Dict[str, str],
) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for parent_label, parent_truth in true_df.groupby(parent_col):
        idx = parent_truth.index
        sub_true = true_df.loc[idx]
        sub_pred = pred_df.loc[idx]
        edge = _edge_breakdown(
            sub_pred,
            sub_true,
            parent_col=parent_col,
            child_col=child_col,
            child_to_parent=child_to_parent,
        )
        rows.append(
            {
                "parent_label": str(parent_label),
                "n_cells": int(len(idx)),
                **edge,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["parent_correct_child_wrong_rate", "n_cells", "parent_label"],
        ascending=[False, False, True],
    )


def _extract_refinement_activation_features(
    *,
    base_result,
    truth_df: pd.DataFrame,
    label_columns: list[str],
    parent_level: str,
    child_level: str,
    child_to_parent: Dict[str, str],
) -> tuple[Dict[str, Any], pd.DataFrame]:
    per_level = evaluate_predictions(base_result.predictions, truth_df[label_columns], label_columns)
    hierarchy_metrics = evaluate_hierarchy_metrics(
        base_result.predictions,
        truth_df[label_columns],
        label_columns,
        hierarchy_rules={child_level: {"parent_col": parent_level, "child_to_parent": child_to_parent}},
    )
    edge = _edge_breakdown(
        base_result.predictions,
        truth_df[label_columns],
        parent_col=parent_level,
        child_col=child_level,
        child_to_parent=child_to_parent,
    )
    subtree_df = _subtree_breakdown(
        base_result.predictions,
        truth_df[label_columns],
        parent_col=parent_level,
        child_col=child_level,
        child_to_parent=child_to_parent,
    )
    subtree_df = subtree_df.copy()
    subtree_df["selection_score"] = (
        pd.to_numeric(subtree_df["parent_correct_child_wrong_rate"], errors="coerce").fillna(0.0)
        * pd.to_numeric(subtree_df["n_cells"], errors="coerce").fillna(0.0)
    )
    total_selection_score = float(subtree_df["selection_score"].sum()) if not subtree_df.empty else 0.0
    top3_score = float(subtree_df.head(3)["selection_score"].sum()) if not subtree_df.empty else 0.0
    finest_col = str(label_columns[-1])
    finest_metrics = dict(per_level.get(finest_col) or {})
    return {
        "n_levels": int(len(label_columns)),
        "finest_macro_f1": float(finest_metrics.get("macro_f1", 0.0)),
        "full_path_accuracy": float(hierarchy_metrics.get("full_path_accuracy", 0.0)),
        "full_path_vs_finest_gap": float(
            finest_metrics.get("macro_f1", 0.0) - hierarchy_metrics.get("full_path_accuracy", 0.0)
        ),
        "parent_correct_child_wrong_rate": float(edge.get("parent_correct_child_wrong_rate", 0.0)),
        "path_break_rate": float(edge.get("path_break_rate", 0.0)),
        "hotspot_concentration_score": (top3_score / total_selection_score) if total_selection_score > 0 else 0.0,
    }, subtree_df


def _extract_activation_features_from_source_run(
    source_run_path: str,
    *,
    label_columns: list[str],
    parent_level: Optional[str],
    child_level: Optional[str],
) -> Dict[str, Any]:
    payload = json.loads(Path(source_run_path).read_text(encoding="utf-8"))
    results = payload.get("results") or []
    if not results:
        raise ValueError("task_weight_policy_source_run does not contain any benchmark results")
    result = results[0]
    if str(result.get("method")) != "atlasmtl":
        raise ValueError("task_weight_policy_source_run must point to an atlasmtl baseline run")
    ablation_config = dict(result.get("ablation_config") or {})
    if str(ablation_config.get("refinement_policy", "none")) != "none":
        raise ValueError("task_weight_policy_source_run must be a baseline run without refinement")
    source_task_weights = [float(x) for x in list(ablation_config.get("task_weights") or [])]
    if source_task_weights and any(abs(x - 1.0) > 1e-8 for x in source_task_weights):
        raise ValueError("task_weight_policy_source_run must use uniform task weights")
    metrics = dict(result.get("metrics") or {})
    if not metrics:
        raise ValueError("task_weight_policy_source_run is missing per-level metrics")
    finest_col = str(label_columns[-1])
    finest_metrics = dict(metrics.get(finest_col) or {})
    coarse_metrics = dict(metrics.get(str(label_columns[0])) or {})
    hierarchy_metrics = dict(result.get("hierarchy_metrics") or {})
    edge_metrics = dict(((hierarchy_metrics.get("edges") or {}).get(str(child_level)) or {})) if child_level else {}
    hotspot_concentration_score = None
    refinement_meta = result.get("refinement_metadata") or {}
    if isinstance(refinement_meta, dict):
        ranking_path = ((refinement_meta.get("artifact_paths") or {}).get("ranking_path"))
        if ranking_path and Path(ranking_path).exists():
            ranking_payload = json.loads(Path(ranking_path).read_text(encoding="utf-8"))
            ranking_rows = list(ranking_payload.get("ranking") or [])
            top3 = sum(float(row.get("selection_score", 0.0)) for row in ranking_rows[:3])
            total = sum(float(row.get("selection_score", 0.0)) for row in ranking_rows)
            hotspot_concentration_score = (top3 / total) if total > 0 else None
    return {
        "finest_macro_f1": float(finest_metrics.get("macro_f1", 0.0)),
        "finest_balanced_accuracy": float(finest_metrics.get("balanced_accuracy", 0.0)),
        "full_path_accuracy": float(hierarchy_metrics.get("full_path_accuracy", 0.0)),
        "coverage": float(finest_metrics.get("coverage", 0.0)),
        "unknown_rate": float(((result.get("behavior_metrics") or {}).get(finest_col) or {}).get("unknown_rate", 0.0)),
        "coarse_to_fine_headroom_gap": float(coarse_metrics.get("macro_f1", 0.0) - finest_metrics.get("macro_f1", 0.0)),
        "parent_correct_child_wrong_rate": float(edge_metrics.get("parent_correct_child_wrong_rate", 0.0)),
        "path_break_rate": float(edge_metrics.get("path_break_rate", 0.0)),
        "hierarchy_on_off_macro_f1_gap": None,
        "hotspot_concentration_score": hotspot_concentration_score,
        "source_run_path": str(Path(source_run_path).resolve()),
    }


def _predict_with_manifest_config(
    model: TrainedModel,
    query_model_input,
    *,
    pred_cfg: Dict[str, Any],
    device: str,
    refinement_config: Optional[Dict[str, Any]] = None,
):
    return predict(
        model,
        query_model_input,
        knn_correction=_normalize_knn_correction(pred_cfg.get("knn_correction")),
        knn_query_obsm_key=pred_cfg.get("knn_query_obsm_key"),
        knn_space=pred_cfg.get("knn_space"),
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
        refinement_config=refinement_config,
        show_progress=False,
        show_summary=False,
    )


def _build_atlasmtl_model(
    *,
    ref_model_input,
    label_columns: list[str],
    coord_targets: Dict[str, Any],
    train_cfg: Dict[str, Any],
    task_weights: list[float],
    device: str,
) -> TrainedModel:
    return build_model(
        adata=ref_model_input,
        label_columns=label_columns,
        coord_targets=coord_targets,
        knn_reference_obsm_key=train_cfg.get("knn_reference_obsm_key"),
        knn_space=train_cfg.get("knn_space"),
        hidden_sizes=train_cfg.get("hidden_sizes"),
        dropout_rate=float(train_cfg.get("dropout_rate", 0.3)),
        batch_size=int(train_cfg.get("batch_size", 256)),
        num_epochs=int(train_cfg.get("num_epochs", 40)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
        optimizer_name=str(train_cfg.get("optimizer_name", "adamw")),
        weight_decay=float(train_cfg.get("weight_decay", 5e-5)),
        scheduler_name=train_cfg.get("scheduler_name"),
        scheduler_factor=float(train_cfg.get("scheduler_factor", 0.5)),
        scheduler_patience=int(train_cfg.get("scheduler_patience", 5)),
        scheduler_min_lr=float(train_cfg.get("scheduler_min_lr", 1e-6)),
        scheduler_monitor=str(train_cfg.get("scheduler_monitor", "val_loss")),
        input_transform=str(train_cfg.get("input_transform", "binary")),
        val_fraction=float(train_cfg.get("val_fraction", 0.0)),
        early_stopping_patience=train_cfg.get("early_stopping_patience"),
        early_stopping_min_delta=float(train_cfg.get("early_stopping_min_delta", 0.0)),
        random_state=int(train_cfg.get("random_state", 42)),
        calibration_method=train_cfg.get("calibration_method"),
        calibration_max_iter=int(train_cfg.get("calibration_max_iter", 100)),
        calibration_lr=float(train_cfg.get("calibration_lr", 0.05)),
        class_weighting=train_cfg.get("class_weighting"),
        class_balanced_sampling=train_cfg.get("class_balanced_sampling"),
        parent_conditioned_child_correction=train_cfg.get("parent_conditioned_child_correction"),
        reference_storage=str(train_cfg.get("reference_storage", "external")),
        reference_path=train_cfg.get("reference_path"),
        init_model_path=train_cfg.get("init_model_path"),
        task_weights=task_weights,
        device=device,
        show_progress=False,
        show_summary=False,
    )


def _apply_refinement_policy(
    *,
    model: TrainedModel,
    ref_model_input,
    query_model_input,
    truth_df: pd.DataFrame,
    label_columns: list[str],
    pred_cfg: Dict[str, Any],
    device: str,
    base_result,
    output_dir: Path,
) -> tuple[Any, Optional[Dict[str, Any]], Dict[str, Optional[str]]]:
    refinement_policy = str(pred_cfg.get("refinement_policy", "none"))
    if refinement_policy == "none":
        return base_result, None, {}
    if refinement_policy != "auto_parent_conditioned_reranker_v1":
        raise ValueError("refinement_policy must be 'none' or 'auto_parent_conditioned_reranker_v1'")

    parent_level, child_level = _infer_parent_child_levels(label_columns, pred_cfg)
    hierarchy_rules = pred_cfg.get("hierarchy_rules")
    if not parent_level or not child_level or not hierarchy_rules or child_level not in hierarchy_rules:
        raise ValueError("auto_parent_conditioned_reranker_v1 requires hierarchy_rules and parent/child levels")

    refinement_dir = output_dir / "refinement"
    refinement_dir.mkdir(parents=True, exist_ok=True)
    refinement_policy_dir = output_dir / "refinement_policy"
    refinement_policy_dir.mkdir(parents=True, exist_ok=True)
    activation_features_path = refinement_policy_dir / "refinement_activation_features.csv"
    activation_decision_path = refinement_policy_dir / "refinement_activation_decision.json"
    ranking_path = refinement_dir / "hotspot_ranking.json"
    per_parent_summary_path = refinement_dir / "per_parent_reranker_summary.csv"
    plan_path = Path(pred_cfg.get("refinement_plan_path")) if pred_cfg.get("refinement_plan_path") else refinement_dir / "refinement_plan.json"
    artifact_path = Path(pred_cfg.get("refinement_artifact_path")) if pred_cfg.get("refinement_artifact_path") else refinement_dir / "parent_conditioned_reranker.pkl"
    comparison_path = refinement_dir / "before_after_comparison.csv"
    edge_path = refinement_dir / "before_after_parent_child_breakdown.csv"
    guardrail_path = refinement_dir / "guardrail_decision.json"

    selection_mode = str(pred_cfg.get("hotspot_selection_mode", "topk"))
    top_k = int(pred_cfg.get("hotspot_top_k", 6))
    cumulative_target = float(pred_cfg.get("hotspot_cumulative_target", 0.6))
    min_cells = int(pred_cfg.get("hotspot_min_cells_per_parent", 200))
    max_selected_parents = int(pred_cfg.get("hotspot_max_selected_parents", 12))
    guardrail_profile_name = str(pred_cfg.get("refinement_guardrail_profile", "phmap_v1"))
    guardrail_profile = get_refinement_guardrail_profile(guardrail_profile_name)
    code_version = _resolve_code_version()
    child_to_parent = dict((hierarchy_rules.get(child_level) or {}).get("child_to_parent") or {})
    activation_features, subtree_df = _extract_refinement_activation_features(
        base_result=base_result,
        truth_df=truth_df,
        label_columns=label_columns,
        parent_level=parent_level,
        child_level=child_level,
        child_to_parent=child_to_parent,
    )
    pd.DataFrame([activation_features]).to_csv(activation_features_path, index=False)
    activation_decision = suggest_parent_conditioned_refinement(
        n_levels=int(activation_features["n_levels"]),
        finest_macro_f1=float(activation_features["finest_macro_f1"]),
        full_path_accuracy=float(activation_features["full_path_accuracy"]),
        parent_correct_child_wrong_rate=float(activation_features["parent_correct_child_wrong_rate"]),
        path_break_rate=float(activation_features["path_break_rate"]),
        hotspot_concentration_score=float(activation_features["hotspot_concentration_score"]),
    )
    activation_decision_path.write_text(
        json.dumps(_json_safe(activation_decision.to_dict()), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not activation_decision.activate_refinement:
        refinement_meta = {
            "policy": refinement_policy,
            "activation_rule_version": activation_decision.activation_rule_version,
            "activated": False,
            "skipped_by_activation_rule": True,
            "guardrail_profile": guardrail_profile_name,
            "guardrail_passed": None,
            "fallback_to_base": False,
            "fallback_reason": "activation_rule_skip",
            "artifact_paths": {
                "activation_features_path": str(activation_features_path),
                "activation_decision_path": str(activation_decision_path),
            },
        }
        return base_result, refinement_meta, refinement_meta["artifact_paths"]

    effective_selection_mode = selection_mode
    effective_top_k = top_k
    effective_cumulative_target = cumulative_target
    if effective_selection_mode == "topk" and "hotspot_top_k" not in pred_cfg:
        effective_top_k = int(activation_decision.recommended_top_k or top_k)
    if effective_selection_mode == "cumulative_contribution" and "hotspot_cumulative_target" not in pred_cfg:
        effective_cumulative_target = float(activation_decision.recommended_cumulative_target or cumulative_target)

    ranking_df, selected_parents, ranking_summary = discover_hotspot_parents(
        subtree_df,
        selection_mode=effective_selection_mode,
        top_k=effective_top_k,
        cumulative_target=effective_cumulative_target,
        min_cells_per_parent=min_cells,
        max_selected_parents=max_selected_parents,
    )
    ranking_payload = {
        "selection_mode": effective_selection_mode,
        "top_k": effective_top_k,
        "cumulative_target": effective_cumulative_target,
        "min_cells_per_parent": min_cells,
        "max_selected_parents": max_selected_parents,
        "selected_parents": selected_parents,
        "summary": ranking_summary,
        "ranking": ranking_df.to_dict(orient="records"),
    }
    ranking_path.write_text(json.dumps(_json_safe(ranking_payload), indent=2, sort_keys=True), encoding="utf-8")
    if not selected_parents:
        refinement_meta = {
            "policy": refinement_policy,
            "activation_rule_version": activation_decision.activation_rule_version,
            "activated": True,
            "skipped_by_activation_rule": False,
            "guardrail_profile": guardrail_profile_name,
            "guardrail_passed": None,
            "fallback_to_base": True,
            "fallback_reason": "no_hotspot_parents_after_filtering",
            "artifact_paths": {
                "activation_features_path": str(activation_features_path),
                "activation_decision_path": str(activation_decision_path),
                "ranking_path": str(ranking_path),
            },
        }
        return base_result, refinement_meta, refinement_meta["artifact_paths"]
    artifact = fit_parent_conditioned_reranker(
        model,
        ref_model_input,
        parent_level=parent_level,
        child_level=child_level,
        hotspot_parents=selected_parents,
        hierarchy_rules=hierarchy_rules,
        batch_size=int(pred_cfg.get("batch_size", 256)),
        device=device,
        selection_metadata={
            "selection_mode": effective_selection_mode,
            "top_k": effective_top_k,
            "cumulative_target": effective_cumulative_target,
            "min_cells_per_parent": min_cells,
            "max_selected_parents": max_selected_parents,
        },
    )
    artifact.save(artifact_path)
    pd.DataFrame(artifact.per_parent_summary).to_csv(per_parent_summary_path, index=False)
    guardrail = {
        "point": "current_manifest + hierarchy_on",
        "name": guardrail_profile["name"],
        "version": guardrail_profile["version"],
        "rules": list(guardrail_profile["rules"]),
        "thresholds": dict(guardrail_profile["thresholds"]),
    }
    plan = build_parent_conditioned_refinement_plan(
        parent_level=parent_level,
        child_level=child_level,
        selection_source="benchmark_runner_base_prediction",
        selection_point="current_manifest + hierarchy_on",
        selection_score="parent_correct_child_wrong_rate * n_cells",
        selection_mode=effective_selection_mode,
        selected_parents=selected_parents,
        artifact_path=str(artifact_path),
        top_k=effective_top_k if effective_selection_mode == "topk" else None,
        cumulative_target=effective_cumulative_target if effective_selection_mode == "cumulative_contribution" else None,
        min_cells_per_parent=min_cells,
        fallback_to_base=True,
        guardrail=guardrail,
        ranking_path=str(ranking_path),
        per_parent_summary_path=str(per_parent_summary_path),
        selection_metadata_version=artifact.selection_metadata_version,
        hierarchy_child_to_parent_hash=artifact.hierarchy_child_to_parent_hash,
        label_space_hash=artifact.label_space_hash,
    )
    plan.save(plan_path)
    refined_result = _predict_with_manifest_config(
        model,
        query_model_input,
        pred_cfg=pred_cfg,
        device=device,
        refinement_config={"method": "auto_parent_conditioned_reranker", "plan_path": str(plan_path)},
    )

    baseline_hierarchy = evaluate_hierarchy_metrics(base_result.predictions, truth_df[label_columns], label_columns, hierarchy_rules=hierarchy_rules)
    refined_hierarchy = evaluate_hierarchy_metrics(refined_result.predictions, truth_df[label_columns], label_columns, hierarchy_rules=hierarchy_rules)
    baseline_metrics = evaluate_predictions(base_result.predictions, truth_df[label_columns], label_columns)[child_level]
    refined_metrics = evaluate_predictions(refined_result.predictions, truth_df[label_columns], label_columns)[child_level]
    baseline_edge = _edge_breakdown(base_result.predictions, truth_df[label_columns], parent_col=parent_level, child_col=child_level, child_to_parent=child_to_parent)
    refined_edge = _edge_breakdown(refined_result.predictions, truth_df[label_columns], parent_col=parent_level, child_col=child_level, child_to_parent=child_to_parent)
    comparison_rows = [
        {
            "variant_name": "baseline",
            "point": "current_manifest",
            f"{child_level}_macro_f1": baseline_metrics["macro_f1"],
            "full_path_accuracy": baseline_hierarchy.get("full_path_accuracy", 0.0),
            "parent_correct_child_wrong_rate": baseline_edge["parent_correct_child_wrong_rate"],
        },
        {
            "variant_name": "refined",
            "point": "current_manifest",
            f"{child_level}_macro_f1": refined_metrics["macro_f1"],
            "full_path_accuracy": refined_hierarchy.get("full_path_accuracy", 0.0),
            "parent_correct_child_wrong_rate": refined_edge["parent_correct_child_wrong_rate"],
        },
    ]
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    pd.DataFrame(
        [
            {"variant_name": "baseline", **baseline_edge},
            {"variant_name": "refined", **refined_edge},
        ]
    ).to_csv(edge_path, index=False)
    guardrail_pass = True
    if guardrail_profile_name == "phmap_v1":
        guardrail_pass = bool(
            float(refined_metrics["macro_f1"]) - float(baseline_metrics["macro_f1"])
            >= float(guardrail_profile["thresholds"]["child_macro_f1_delta_min"])
            and float(refined_hierarchy.get("full_path_accuracy", 0.0)) - float(baseline_hierarchy.get("full_path_accuracy", 0.0))
            >= float(guardrail_profile["thresholds"]["full_path_accuracy_delta_min"])
            and float(refined_edge["parent_correct_child_wrong_rate"]) - float(baseline_edge["parent_correct_child_wrong_rate"])
            <= float(guardrail_profile["thresholds"]["parent_correct_child_wrong_rate_delta_max"])
        )
    guardrail_payload = {
        "passed": bool(guardrail_pass),
        "guardrail_profile_name": guardrail_profile["name"],
        "guardrail_profile_version": guardrail_profile["version"],
        "guardrail_rules_expanded": list(guardrail_profile["rules"]),
        "thresholds": dict(guardrail_profile["thresholds"]),
        "code_version": code_version,
        "baseline": comparison_rows[0],
        "candidate": comparison_rows[1],
        "selected_parents": selected_parents,
    }
    guardrail_path.write_text(json.dumps(_json_safe(guardrail_payload), indent=2, sort_keys=True), encoding="utf-8")

    refinement_meta = {
        "policy": refinement_policy,
        "activation_rule_version": activation_decision.activation_rule_version,
        "activated": True,
        "skipped_by_activation_rule": False,
        "parent_level": parent_level,
        "child_level": child_level,
        "selected_parents": selected_parents,
        "guardrail_profile": guardrail_profile_name,
        "guardrail_passed": bool(guardrail_pass),
        "fallback_to_base": bool(not guardrail_pass),
        "fallback_reason": None if guardrail_pass else "guardrail_failed",
        "artifact_paths": {
            "activation_features_path": str(activation_features_path),
            "activation_decision_path": str(activation_decision_path),
            "ranking_path": str(ranking_path),
            "plan_path": str(plan_path),
            "artifact_path": str(artifact_path),
            "per_parent_summary_path": str(per_parent_summary_path),
            "comparison_path": str(comparison_path),
            "edge_path": str(edge_path),
            "guardrail_path": str(guardrail_path),
        },
    }
    final_result = refined_result if guardrail_pass else base_result
    return final_result, refinement_meta, refinement_meta["artifact_paths"]


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
    atlasmtl_layer_cfg = resolve_atlasmtl_layer_config(manifest)
    task_weights = resolve_task_weights(manifest, label_columns)
    protocol_context = {
        "protocol_version": int(manifest.get("protocol_version", PROTOCOL_VERSION)),
        "random_seed": manifest.get("random_seed"),
        "split_name": manifest.get("split_name"),
        "split_description": manifest.get("split_description"),
        "reference_subset": manifest.get("reference_subset"),
        "query_subset": manifest.get("query_subset"),
        "domain_key": manifest.get("domain_key"),
    }
    ref_model_input = adata_with_matrix_from_layer(ref, layer_name=atlasmtl_layer_cfg["reference_layer"])
    query_model_input = adata_with_matrix_from_layer(query, layer_name=atlasmtl_layer_cfg["query_layer"])
    truth_df = query.obs.loc[:, label_columns].astype(str).copy()
    parent_level, child_level = _infer_parent_child_levels(label_columns, dict(manifest.get("predict") or {}))
    refinement_artifacts: Dict[str, Optional[str]] = {}
    refinement_metadata: Optional[Dict[str, Any]] = None
    task_weight_policy_metadata: Optional[Dict[str, Any]] = None
    selector_artifacts: Dict[str, Optional[str]] = {}
    policy_dag: Dict[str, Any] = {
        "weight_policy_stage": {"enabled": False},
        "refinement_stage": {"enabled": False},
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
        weight_policy = str(train_cfg.get("task_weight_policy", "manual"))
        selector_name = str(train_cfg.get("task_weight_selector", "none"))
        if weight_policy not in {"manual", "activation_rule_v1"}:
            raise ValueError("task_weight_policy must be 'manual' or 'activation_rule_v1'")
        if selector_name not in {"none", "candidate_selector_v1"}:
            raise ValueError("task_weight_selector must be 'none' or 'candidate_selector_v1'")
        chosen_task_weights = list(task_weights)
        chosen_schedule_name = task_weight_scheme_name(chosen_task_weights)
        model = None
        train_cfg_used = dict(train_cfg)
        if weight_policy == "activation_rule_v1":
            source_run_path = train_cfg.get("task_weight_policy_source_run")
            if not source_run_path:
                raise ValueError("task_weight_policy='activation_rule_v1' requires task_weight_policy_source_run")
            features = _extract_activation_features_from_source_run(
                str(source_run_path),
                label_columns=label_columns,
                parent_level=parent_level,
                child_level=child_level,
            )
            decision = suggest_task_weight_schedule(
                n_levels=len(label_columns),
                finest_macro_f1=float(features["finest_macro_f1"]),
                finest_balanced_accuracy=float(features["finest_balanced_accuracy"]),
                full_path_accuracy=float(features["full_path_accuracy"]),
                parent_correct_child_wrong_rate=float(features["parent_correct_child_wrong_rate"]),
                coarse_to_fine_headroom_gap=features.get("coarse_to_fine_headroom_gap"),
                hierarchy_on_off_macro_f1_gap=features.get("hierarchy_on_off_macro_f1_gap"),
                hotspot_concentration_score=features.get("hotspot_concentration_score"),
            )
            policy_dir = output_dir / "weight_policy"
            policy_dir.mkdir(parents=True, exist_ok=True)
            features_path = policy_dir / "weight_activation_features.csv"
            pd.DataFrame([features]).to_csv(features_path, index=False)
            decision_path = policy_dir / "weight_activation_decision.json"
            decision_payload = {
                **decision.to_dict(),
                "policy_name": weight_policy,
                "policy_version": decision.activation_rule_version,
                "source_run_path": str(Path(str(source_run_path)).resolve()),
                "code_version": _resolve_code_version(),
            }
            decision_path.write_text(json.dumps(_json_safe(decision_payload), indent=2, sort_keys=True), encoding="utf-8")
            task_weight_policy_metadata = {
                "policy": weight_policy,
                "selector": selector_name,
                "source_run_path": str(Path(str(source_run_path)).resolve()),
                "decision": decision_payload,
                "decision_path": str(decision_path),
                "features_path": str(features_path),
            }
            policy_dag["weight_policy_stage"] = {
                "enabled": True,
                "source_run_path": str(Path(str(source_run_path)).resolve()),
                "policy_name": weight_policy,
                "policy_version": decision.activation_rule_version,
                "selector_name": selector_name,
                "activate_nonuniform_weighting": bool(decision.activate_nonuniform_weighting),
            }
            if decision.activate_nonuniform_weighting and selector_name == "candidate_selector_v1":
                candidates = resolve_task_weight_candidates(
                    manifest,
                    candidate_schedules=decision.candidate_schedules,
                    label_columns=label_columns,
                )
                selector_rows: list[Dict[str, Any]] = []
                best_result = None
                best_model = None
                best_artifact_paths = None
                selector_dir = policy_dir / "selector"
                selector_dir.mkdir(parents=True, exist_ok=True)
                for schedule_name, schedule_weights in candidates.items():
                    candidate_model = _build_atlasmtl_model(
                        ref_model_input=ref_model_input,
                        label_columns=label_columns,
                        coord_targets=coord_targets,
                        train_cfg=train_cfg,
                        task_weights=schedule_weights,
                        device=device,
                    )
                    candidate_result = _predict_with_manifest_config(
                        candidate_model,
                        query_model_input,
                        pred_cfg=manifest.get("predict") or {},
                        device=device,
                    )
                    candidate_level_metrics = evaluate_predictions(candidate_result.predictions, truth_df, label_columns)[label_columns[-1]]
                    candidate_hierarchy = evaluate_hierarchy_metrics(
                        candidate_result.predictions,
                        truth_df,
                        label_columns,
                        hierarchy_rules=(manifest.get("predict") or {}).get("hierarchy_rules"),
                    ) if (manifest.get("predict") or {}).get("hierarchy_rules") else {}
                    candidate_edge = (
                        _edge_breakdown(
                            candidate_result.predictions,
                            truth_df,
                            parent_col=parent_level,
                            child_col=child_level,
                            child_to_parent=dict((((manifest.get("predict") or {}).get("hierarchy_rules") or {}).get(child_level) or {}).get("child_to_parent") or {}),
                        )
                        if parent_level and child_level and (manifest.get("predict") or {}).get("hierarchy_rules")
                        else {"parent_correct_child_wrong_rate": 0.0}
                    )
                    selector_rows.append(
                        {
                            "schedule_name": str(schedule_name),
                            "task_weights": [float(x) for x in schedule_weights],
                            "finest_macro_f1": float(candidate_level_metrics["macro_f1"]),
                            "full_path_accuracy": float(candidate_hierarchy.get("full_path_accuracy", 0.0)),
                            "parent_correct_child_wrong_rate": float(candidate_edge["parent_correct_child_wrong_rate"]),
                        }
                    )
                    if best_model is None:
                        best_model = candidate_model
                        best_result = candidate_result
                selector_choice = select_task_weight_candidate_from_summary(selector_rows)
                chosen_schedule_name = str(selector_choice["schedule_name"])
                chosen_task_weights = [float(x) for x in selector_choice["task_weights"]]
                comparison_path = selector_dir / "weight_selector_comparison.csv"
                pd.DataFrame(selector_rows).to_csv(comparison_path, index=False)
                ranked_candidates = sorted(
                    selector_rows,
                    key=lambda row: (
                        -float(row["full_path_accuracy"]),
                        float(row["parent_correct_child_wrong_rate"]),
                        -float(row["finest_macro_f1"]),
                        0
                        if str(row["schedule_name"]).startswith("mild")
                        else 1 if str(row["schedule_name"]).startswith("strong") else 2,
                        str(row["schedule_name"]),
                    ),
                )
                ranked_path = selector_dir / "weight_selector_candidates_ranked.json"
                ranked_path.write_text(
                    json.dumps(
                        _json_safe(
                            {
                                "selector_name": selector_name,
                                "selector_version": "candidate_selector_v1",
                                "ranking_rule": [
                                    "full_path_accuracy desc",
                                    "parent_correct_child_wrong_rate asc",
                                    "finest_macro_f1 desc",
                                    "prefer mild over strong over uniform on ties",
                                ],
                                "tie_break_rule": "prefer mild > strong > uniform for near ties",
                                "ranked_candidates": ranked_candidates,
                                "selected_candidate": selector_choice,
                            }
                        ),
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                selector_decision_path = selector_dir / "weight_selector_decision.json"
                selector_decision_path.write_text(json.dumps(_json_safe(selector_choice), indent=2, sort_keys=True), encoding="utf-8")
                selector_artifacts = {
                    "weight_selector_comparison_path": str(comparison_path),
                    "weight_selector_decision_path": str(selector_decision_path),
                    "weight_selector_candidates_ranked_path": str(ranked_path),
                }
                task_weight_policy_metadata["selector_decision"] = selector_choice
                task_weight_policy_metadata["selector_artifacts"] = selector_artifacts
                policy_dag["weight_policy_stage"]["selector_selected_schedule"] = str(chosen_schedule_name)
                model = _build_atlasmtl_model(
                    ref_model_input=ref_model_input,
                    label_columns=label_columns,
                    coord_targets=coord_targets,
                    train_cfg=train_cfg,
                    task_weights=chosen_task_weights,
                    device=device,
                )
            elif decision.activate_nonuniform_weighting:
                chosen_schedule_name = "uniform_pending_candidate_test"
                chosen_task_weights = list(task_weights)
                model = _build_atlasmtl_model(
                    ref_model_input=ref_model_input,
                    label_columns=label_columns,
                    coord_targets=coord_targets,
                    train_cfg=train_cfg,
                    task_weights=chosen_task_weights,
                    device=device,
                )
            else:
                chosen_task_weights = list(decision.recommended_schedule or [1.0 for _ in label_columns])
                chosen_schedule_name = str(decision.recommended_schedule_name)
                model = _build_atlasmtl_model(
                    ref_model_input=ref_model_input,
                    label_columns=label_columns,
                    coord_targets=coord_targets,
                    train_cfg=train_cfg,
                    task_weights=chosen_task_weights,
                    device=device,
                )
        else:
            model = _build_atlasmtl_model(
                ref_model_input=ref_model_input,
                label_columns=label_columns,
                coord_targets=coord_targets,
                train_cfg=train_cfg,
                task_weights=chosen_task_weights,
                device=device,
            )
        model_out = output_dir / "atlasmtl_model.pth"
        model.save(str(model_out))
        artifact_paths = resolve_manifest_paths(str(output_dir / "atlasmtl_model_manifest.json"))
        model_input_path = str(model_out.resolve())
        model_source = "trained_in_runner"
        task_weights = chosen_task_weights
        train_cfg_used = dict(train_cfg, task_weights=task_weights)
        train_cfg_used["task_weight_scheme"] = chosen_schedule_name
        policy_dag["weight_policy_stage"]["selected_schedule_name"] = str(chosen_schedule_name)
        policy_dag["weight_policy_stage"]["selected_task_weights"] = [float(x) for x in task_weights]

    pred_cfg = dict(manifest.get("predict") or {})
    base_result = _predict_with_manifest_config(
        model,
        query_model_input,
        pred_cfg=pred_cfg,
        device=device,
    )
    result, refinement_metadata, refinement_artifacts = _apply_refinement_policy(
        model=model,
        ref_model_input=ref_model_input,
        query_model_input=query_model_input,
        truth_df=truth_df,
        label_columns=label_columns,
        pred_cfg=pred_cfg,
        device=device,
        base_result=base_result,
        output_dir=output_dir,
    )
    policy_dag["refinement_stage"] = {
        "enabled": bool(str(pred_cfg.get("refinement_policy", "none")) != "none"),
        "policy_name": str(pred_cfg.get("refinement_policy", "none")),
        "source_prediction": "chosen_base_model_prediction",
        "activation_rule_skipped": bool((refinement_metadata or {}).get("skipped_by_activation_rule", False)),
        "fallback_to_base": bool((refinement_metadata or {}).get("fallback_to_base", False)),
        "guardrail_passed": (refinement_metadata or {}).get("guardrail_passed"),
    }

    per_level = evaluate_predictions(result.predictions, truth_df, label_columns)
    behavior = evaluate_prediction_behavior(result.predictions, truth_df, label_columns)
    by_domain = None
    behavior_by_domain = None
    domain_key = manifest.get("domain_key")
    if domain_key and domain_key in query.obs.columns:
        by_domain = evaluate_predictions_by_group(
            result.predictions,
            truth_df,
            label_columns,
            group=query.obs[domain_key],
        )
        behavior_by_domain = evaluate_prediction_behavior_by_group(
            result.predictions,
            truth_df,
            label_columns,
            group=query.obs[domain_key],
        )

    train_usage = model.get_resource_usage()
    pred_usage = result.get_resource_usage()
    artifacts = _artifact_sizes_mb(artifact_paths) if artifact_paths else {}
    preprocess_cfg = ((manifest.get("preprocess") or {}).get("config") or {}) if isinstance(manifest.get("preprocess"), dict) else {}
    feature_space = preprocess_cfg.get("feature_space") or manifest.get("feature_space")
    n_top_genes = preprocess_cfg.get("n_top_genes")
    input_transform = str(train_cfg.get("input_transform", "binary"))
    if str(feature_space) == "hvg" and n_top_genes:
        feature_token = f"hvg{n_top_genes}"
    else:
        feature_token = str(feature_space or "whole")
    variant_tokens = [
        str(device),
        feature_token,
        str(input_transform),
        task_weight_scheme_name(task_weights),
    ]
    variant_name = "atlasmtl_" + "_".join(str(token) for token in variant_tokens if token)
    ablation_config = {
        "variant_name": variant_name,
        "device": str(device),
        "reference_matrix_source": matrix_source_label(atlasmtl_layer_cfg["reference_layer"]),
        "query_matrix_source": matrix_source_label(atlasmtl_layer_cfg["query_layer"]),
        "counts_layer": atlasmtl_layer_cfg["counts_layer"],
        "input_transform": input_transform,
        "feature_space": feature_space,
        "n_top_genes": n_top_genes,
        "task_weights": task_weights,
        "task_weight_scheme": task_weight_scheme_name(task_weights),
        "task_weight_policy": str(train_cfg.get("task_weight_policy", "manual")),
        "task_weight_selector": str(train_cfg.get("task_weight_selector", "none")),
        "refinement_policy": str(pred_cfg.get("refinement_policy", "none")),
    }

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
            truth_df,
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
        "refinement_artifacts": refinement_artifacts or None,
        "model_source": model_source,
        "model_input_path": model_input_path,
        "variant_name": variant_name,
        "ablation_config": ablation_config,
        "input_contract": build_input_contract(
            reference_matrix_source=matrix_source_label(atlasmtl_layer_cfg["reference_layer"]),
            query_matrix_source=matrix_source_label(atlasmtl_layer_cfg["query_layer"]),
            counts_layer=((manifest.get("preprocess") or {}).get("config") or {}).get("counts_layer")
            if isinstance(manifest.get("preprocess"), dict)
            else manifest.get("counts_layer"),
            feature_alignment="reference_feature_panel_exact_order",
            normalization_mode=f"atlasmtl_extract_matrix:{input_transform}",
            label_scope="multi_level",
            backend="atlasmtl",
        ),
        "train_config_used": (train_cfg_used if not atlasmtl_model else None),
        "predict_config_used": pred_cfg,
        "prediction_metadata": result.metadata,
        "refinement_metadata": refinement_metadata,
        "task_weight_policy_metadata": task_weight_policy_metadata,
        "policy_dag": policy_dag,
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
                "variant_name": _effective_variant_name(result),
                "dataset_name": result.get("dataset_name"),
                "level": level,
                "split_name": (result.get("protocol_context") or {}).get("split_name"),
                "model_source": result.get("model_source"),
                "backend": ((result.get("input_contract") or {}).get("backend")),
                "reference_matrix_source": ((result.get("input_contract") or {}).get("reference_matrix_source")),
                "query_matrix_source": ((result.get("input_contract") or {}).get("query_matrix_source")),
                "counts_layer": ((result.get("input_contract") or {}).get("counts_layer")),
                "normalization_mode": ((result.get("input_contract") or {}).get("normalization_mode")),
                "feature_alignment": ((result.get("input_contract") or {}).get("feature_alignment")),
                "label_scope": ((result.get("input_contract") or {}).get("label_scope")),
                "knn_vote_mode": (result.get("predict_config_used") or {}).get("knn_vote_mode", "majority"),
                "knn_reference_mode": (result.get("predict_config_used") or {}).get("knn_reference_mode", "full"),
                "knn_index_mode": (result.get("predict_config_used") or {}).get("knn_index_mode", "exact"),
                "enforce_hierarchy": bool((result.get("predict_config_used") or {}).get("enforce_hierarchy", False)),
                "openset_method": (result.get("predict_config_used") or {}).get("openset_method"),
                "task_weight_policy": ((result.get("ablation_config") or {}).get("task_weight_policy")),
                "task_weight_selector": ((result.get("ablation_config") or {}).get("task_weight_selector")),
                "refinement_policy": ((result.get("ablation_config") or {}).get("refinement_policy")),
                "refinement_guardrail_passed": ((result.get("refinement_metadata") or {}).get("guardrail_passed")),
                "refinement_fallback_to_base": ((result.get("refinement_metadata") or {}).get("fallback_to_base")),
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
                    "variant_name": _effective_variant_name(result),
                    "dataset_name": result.get("dataset_name"),
                    "domain": domain,
                    "level": level,
                    "split_name": (result.get("protocol_context") or {}).get("split_name"),
                    "backend": ((result.get("input_contract") or {}).get("backend")),
                    "reference_matrix_source": ((result.get("input_contract") or {}).get("reference_matrix_source")),
                    "query_matrix_source": ((result.get("input_contract") or {}).get("query_matrix_source")),
                    "counts_layer": ((result.get("input_contract") or {}).get("counts_layer")),
                    "knn_vote_mode": (result.get("predict_config_used") or {}).get("knn_vote_mode", "majority"),
                    "knn_reference_mode": (result.get("predict_config_used") or {}).get("knn_reference_mode", "full"),
                    "knn_index_mode": (result.get("predict_config_used") or {}).get("knn_index_mode", "exact"),
                    "enforce_hierarchy": bool((result.get("predict_config_used") or {}).get("enforce_hierarchy", False)),
                    "refinement_policy": ((result.get("ablation_config") or {}).get("refinement_policy")),
                    "refinement_guardrail_passed": ((result.get("refinement_metadata") or {}).get("guardrail_passed")),
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
