#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from anndata import read_h5ad

from atlasmtl import TrainedModel, predict
from benchmark.methods.atlasmtl_inputs import adata_with_matrix_from_layer, resolve_atlasmtl_layer_config


REPO_ROOT = Path(__file__).resolve().parents[4]
DOSSIER_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-09_phmap_study_split_validation"
RESULTS_DIR = DOSSIER_ROOT / "results_summary"
MANIFEST_ROOT = DOSSIER_ROOT / "manifests" / "phase3_tradeoff"
TMP_ROOT = Path("/tmp/atlasmtl_benchmarks/2026-03-10/phmap_study_split_phase3_tradeoff")
FINEST_LEVEL = "anno_lv4"
LABEL_COLUMNS = ["anno_lv1", "anno_lv2", "anno_lv3", "anno_lv4"]
UNKNOWN = "Unknown"


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _first_result(metrics_payload: Dict[str, Any]) -> Dict[str, Any]:
    return ((metrics_payload.get("results") or [None])[0] or {})


def _runtime_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    train_usage = result.get("train_usage") or {}
    predict_usage = result.get("predict_usage") or {}
    return {
        "train_elapsed_seconds": train_usage.get("elapsed_seconds"),
        "predict_elapsed_seconds": predict_usage.get("elapsed_seconds"),
        "train_process_peak_rss_gb": train_usage.get("process_peak_rss_gb"),
        "predict_process_peak_rss_gb": predict_usage.get("process_peak_rss_gb"),
        "train_gpu_peak_memory_gb": train_usage.get("gpu_peak_memory_gb"),
        "predict_gpu_peak_memory_gb": predict_usage.get("gpu_peak_memory_gb"),
    }


def _load_manifest(path: Path) -> Dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _manifest_for(config_name: str, point: str, hierarchy_setting: str) -> Path:
    return MANIFEST_ROOT / "predict" / config_name / point / f"hierarchy_{hierarchy_setting}" / "atlasmtl_phase3_predict.yaml"


def _load_train_rows() -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    train_root = TMP_ROOT / "train"
    if not train_root.exists():
        return rows
    for config_dir in sorted(p for p in train_root.iterdir() if p.is_dir()):
        run_dir = config_dir / "runs" / "atlasmtl"
        metrics_payload = _safe_read_json(run_dir / "metrics.json")
        if metrics_payload is None:
            continue
        result = _first_result(metrics_payload)
        runtime = _runtime_fields(result)
        rows[config_dir.name] = {
            "train_elapsed_seconds": runtime["train_elapsed_seconds"],
            "train_process_peak_rss_gb": runtime["train_process_peak_rss_gb"],
            "train_gpu_peak_memory_gb": runtime["train_gpu_peak_memory_gb"],
            "model_manifest_path": (run_dir / "atlasmtl_model_manifest.json").as_posix(),
        }
    return rows


def _scan_predict_rows(train_rows: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    predict_root = TMP_ROOT / "predict"
    if not predict_root.exists():
        return rows
    for config_dir in sorted(p for p in predict_root.iterdir() if p.is_dir()):
        for point_dir in sorted(p for p in config_dir.iterdir() if p.is_dir()):
            for hierarchy_dir in sorted(p for p in point_dir.iterdir() if p.is_dir()):
                run_dir = hierarchy_dir / "runs" / "atlasmtl"
                metrics_payload = _safe_read_json(run_dir / "metrics.json")
                if metrics_payload is None:
                    continue
                result = _first_result(metrics_payload)
                runtime = _runtime_fields(result)
                train_info = train_rows.get(config_dir.name, {})
                hierarchy_setting = hierarchy_dir.name.replace("hierarchy_", "")
                rows.append(
                    {
                        "dataset": "PHMap_Lung_Full_v43_light",
                        "track": "gpu",
                        "config_name": config_dir.name,
                        "point": point_dir.name,
                        "hierarchy_setting": hierarchy_setting,
                        "enforce_hierarchy": hierarchy_setting == "on",
                        "run_dir": run_dir,
                        "result": result,
                        "train_elapsed_seconds": train_info.get("train_elapsed_seconds"),
                        "train_process_peak_rss_gb": train_info.get("train_process_peak_rss_gb"),
                        "train_gpu_peak_memory_gb": train_info.get("train_gpu_peak_memory_gb"),
                        "predict_elapsed_seconds": runtime["predict_elapsed_seconds"],
                        "predict_process_peak_rss_gb": runtime["predict_process_peak_rss_gb"],
                        "predict_gpu_peak_memory_gb": runtime["predict_gpu_peak_memory_gb"],
                    }
                )
    return rows


def _levelwise_rows(run_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in run_rows:
        metrics = (run["result"].get("metrics") or {})
        behavior = (run["result"].get("behavior_metrics") or {})
        for level, metric_payload in metrics.items():
            behavior_payload = behavior.get(level) or {}
            rows.append(
                {
                    "dataset": run["dataset"],
                    "track": run["track"],
                    "config_name": run["config_name"],
                    "point": run["point"],
                    "hierarchy_setting": run["hierarchy_setting"],
                    "enforce_hierarchy": run["enforce_hierarchy"],
                    "level": level,
                    "accuracy": metric_payload.get("accuracy"),
                    "macro_f1": metric_payload.get("macro_f1"),
                    "balanced_accuracy": metric_payload.get("balanced_accuracy"),
                    "coverage": metric_payload.get("coverage"),
                    "covered_accuracy": metric_payload.get("covered_accuracy"),
                    "unknown_rate": behavior_payload.get("unknown_rate"),
                    "train_elapsed_seconds": run["train_elapsed_seconds"],
                    "predict_elapsed_seconds": run["predict_elapsed_seconds"],
                }
            )
    return rows


def _hierarchy_rows(run_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in run_rows:
        hierarchy = (run["result"].get("hierarchy_metrics") or {})
        edges = hierarchy.get("edges") or {}
        edge_rates = {child_col: (payload or {}).get("path_consistency_rate") for child_col, payload in edges.items()}
        valid_rates = [float(v) for v in edge_rates.values() if v is not None]
        rows.append(
            {
                "dataset": run["dataset"],
                "track": run["track"],
                "config_name": run["config_name"],
                "point": run["point"],
                "hierarchy_setting": run["hierarchy_setting"],
                "enforce_hierarchy": run["enforce_hierarchy"],
                "full_path_accuracy": hierarchy.get("full_path_accuracy"),
                "full_path_coverage": hierarchy.get("full_path_coverage"),
                "full_path_covered_accuracy": hierarchy.get("full_path_covered_accuracy"),
                "mean_path_consistency_rate": sum(valid_rates) / len(valid_rates) if valid_rates else None,
                "min_path_consistency_rate": min(valid_rates) if valid_rates else None,
            }
        )
    return rows


def _replay_predictions(run: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    manifest_path = _manifest_for(run["config_name"], run["point"], run["hierarchy_setting"])
    manifest = _load_manifest(manifest_path)
    layer_cfg = resolve_atlasmtl_layer_config(manifest)
    query = read_h5ad(str(manifest["query_h5ad"]))
    query_model_input = adata_with_matrix_from_layer(query, layer_name=layer_cfg["query_layer"])
    model_manifest = TMP_ROOT / "train" / run["config_name"] / "runs" / "atlasmtl" / "atlasmtl_model_manifest.json"
    model = TrainedModel.load(str(model_manifest))
    pred_cfg = dict(manifest.get("predict") or {})
    result = predict(
        model,
        query_model_input,
        knn_correction=str(pred_cfg.get("knn_correction", "off")),
        batch_size=int(pred_cfg.get("batch_size", 512)),
        input_transform=pred_cfg.get("input_transform"),
        hierarchy_rules=pred_cfg.get("hierarchy_rules"),
        enforce_hierarchy=bool(pred_cfg.get("enforce_hierarchy", False)),
        device="cuda",
        show_progress=False,
        show_summary=False,
    )
    true_df = query.obs.loc[:, LABEL_COLUMNS + ["study"]].copy()
    return result.predictions.copy(), true_df, pred_cfg.get("hierarchy_rules") or {}


def _accuracy(true: pd.Series, pred: pd.Series) -> float:
    return float((true.astype(str) == pred.astype(str)).mean()) if len(true) else 0.0


def _edge_breakdown(
    pred_df: pd.DataFrame,
    true_df: pd.DataFrame,
    *,
    parent_col: str,
    child_col: str,
    child_to_parent: Dict[str, str],
) -> Dict[str, float]:
    parent_true = true_df[parent_col].astype(str)
    child_true = true_df[child_col].astype(str)
    parent_pred = pred_df[f"pred_{parent_col}"].astype(str)
    child_pred = pred_df[f"pred_{child_col}"].astype(str)
    implied_parent = child_pred.map(lambda x: child_to_parent.get(x) if x != UNKNOWN else UNKNOWN).fillna("MISSING_PARENT")
    parent_correct = parent_true == parent_pred
    child_correct = child_true == child_pred
    path_break = (child_pred != UNKNOWN) & (parent_pred != UNKNOWN) & (implied_parent != parent_pred)
    return {
        "parent_accuracy": _accuracy(parent_true, parent_pred),
        "child_accuracy": _accuracy(child_true, child_pred),
        "parent_correct_child_wrong_rate": float((parent_correct & ~child_correct).mean()),
        "parent_wrong_child_correct_rate": float((~parent_correct & child_correct).mean()),
        "both_wrong_rate": float((~parent_correct & ~child_correct).mean()),
        "child_unknown_rate": float((child_pred == UNKNOWN).mean()),
        "parent_unknown_rate": float((parent_pred == UNKNOWN).mean()),
        "path_break_rate": float(path_break.mean()),
    }


def _parent_child_rows(run_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in run_rows:
        pred_df, true_df, hierarchy_rules = _replay_predictions(run)
        for child_col, rule in hierarchy_rules.items():
            parent_col = str(rule["parent_col"])
            child_to_parent = {str(k): str(v) for k, v in dict(rule["child_to_parent"]).items()}
            edge_metrics = _edge_breakdown(
                pred_df,
                true_df,
                parent_col=parent_col,
                child_col=str(child_col),
                child_to_parent=child_to_parent,
            )
            rows.append(
                {
                    "config_name": run["config_name"],
                    "point": run["point"],
                    "hierarchy_setting": run["hierarchy_setting"],
                    "child_col": str(child_col),
                    "parent_col": parent_col,
                    "path_consistency_rate": ((run["result"].get("hierarchy_metrics") or {}).get("edges") or {}).get(str(child_col), {}).get("path_consistency_rate"),
                    **edge_metrics,
                }
            )
    return rows


def _subtree_rows(run_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    child_col = "anno_lv4"
    parent_col = "anno_lv3"
    for run in run_rows:
        pred_df, true_df, hierarchy_rules = _replay_predictions(run)
        child_to_parent = {
            str(k): str(v)
            for k, v in dict((hierarchy_rules.get(child_col) or {}).get("child_to_parent") or {}).items()
        }
        parent_true = true_df[parent_col].astype(str)
        child_true = true_df[child_col].astype(str)
        parent_pred = pred_df[f"pred_{parent_col}"].astype(str)
        child_pred = pred_df[f"pred_{child_col}"].astype(str)
        implied_parent = child_pred.map(lambda x: child_to_parent.get(x) if x != UNKNOWN else UNKNOWN).fillna("MISSING_PARENT")
        grouped = true_df.groupby(parent_col, observed=False).groups
        for parent_label, idx in grouped.items():
            idx = list(idx)
            pt = parent_true.loc[idx]
            ct = child_true.loc[idx]
            pp = parent_pred.loc[idx]
            cp = child_pred.loc[idx]
            ip = implied_parent.loc[idx]
            parent_correct = pt == pp
            child_correct = ct == cp
            path_break = (cp != UNKNOWN) & (pp != UNKNOWN) & (ip != pp)
            rows.append(
                {
                    "config_name": run["config_name"],
                    "point": run["point"],
                    "hierarchy_setting": run["hierarchy_setting"],
                    "parent_label": str(parent_label),
                    "n_cells": int(len(idx)),
                    "lv4_accuracy": _accuracy(ct, cp),
                    "lv4_unknown_rate": float((cp == UNKNOWN).mean()),
                    "parent_correct_child_wrong_rate": float((parent_correct & ~child_correct).mean()),
                    "path_break_rate": float(path_break.mean()),
                }
            )
    return rows


def _by_study_rows(run_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from atlasmtl.core.evaluate import evaluate_hierarchy_metrics, evaluate_prediction_behavior, evaluate_predictions

    rows: List[Dict[str, Any]] = []
    for run in run_rows:
        pred_df, true_df, hierarchy_rules = _replay_predictions(run)
        for study, idx in true_df.groupby("study", observed=False).groups.items():
            idx = list(idx)
            sub_pred = pred_df.loc[idx]
            sub_true = true_df.loc[idx, LABEL_COLUMNS]
            metrics = evaluate_predictions(sub_pred, sub_true, LABEL_COLUMNS)
            behavior = evaluate_prediction_behavior(sub_pred, sub_true, LABEL_COLUMNS)
            hierarchy = evaluate_hierarchy_metrics(sub_pred, sub_true, LABEL_COLUMNS, hierarchy_rules=hierarchy_rules)
            rows.append(
                {
                    "config_name": run["config_name"],
                    "point": run["point"],
                    "hierarchy_setting": run["hierarchy_setting"],
                    "study": str(study),
                    "anno_lv4_macro_f1": (metrics.get(FINEST_LEVEL) or {}).get("macro_f1"),
                    "coverage": (metrics.get(FINEST_LEVEL) or {}).get("coverage"),
                    "unknown_rate": (behavior.get(FINEST_LEVEL) or {}).get("unknown_rate"),
                    "full_path_accuracy": hierarchy.get("full_path_accuracy"),
                }
            )
    return rows


def _write_markdown(
    finest: pd.DataFrame,
    hierarchy: pd.DataFrame,
    edge_df: pd.DataFrame,
    subtree_df: pd.DataFrame,
    study_df: pd.DataFrame,
) -> None:
    def _table(df: pd.DataFrame, cols: List[str]) -> str:
        return df[cols].to_markdown(index=False) if len(df) else "No rows found."

    summary_lines = [
        "# PH-Map Study-Split Phase 3 Tradeoff Attribution",
        "",
        "## Finest-Level Rows",
        "",
        _table(
            finest,
            ["config_name", "point", "hierarchy_setting", "macro_f1", "balanced_accuracy", "accuracy", "coverage", "unknown_rate"],
        ),
        "",
        "## Hierarchy Rows",
        "",
        _table(
            hierarchy,
            ["config_name", "point", "hierarchy_setting", "full_path_accuracy", "full_path_coverage", "mean_path_consistency_rate"],
        ),
        "",
        "## Parent-Child Breakdown",
        "",
        _table(
            edge_df,
            [
                "config_name",
                "point",
                "hierarchy_setting",
                "child_col",
                "parent_col",
                "path_consistency_rate",
                "parent_correct_child_wrong_rate",
                "parent_wrong_child_correct_rate",
                "both_wrong_rate",
                "path_break_rate",
            ],
        ),
        "",
        "## Subtree Hotspots",
        "",
        _table(
            subtree_df.sort_values(["config_name", "point", "hierarchy_setting", "path_break_rate"], ascending=[True, True, True, False]).head(16),
            ["config_name", "point", "hierarchy_setting", "parent_label", "n_cells", "lv4_accuracy", "lv4_unknown_rate", "parent_correct_child_wrong_rate", "path_break_rate"],
        ),
        "",
        "## Study Breakdown",
        "",
        _table(
            study_df,
            ["config_name", "point", "hierarchy_setting", "study", "anno_lv4_macro_f1", "coverage", "unknown_rate", "full_path_accuracy"],
        ),
        "",
    ]
    (RESULTS_DIR / "phase3_tradeoff_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    train_rows = _load_train_rows()
    run_rows = _scan_predict_rows(train_rows)
    levelwise = pd.DataFrame(_levelwise_rows(run_rows))
    hierarchy = pd.DataFrame(_hierarchy_rows(run_rows))
    finest = levelwise.loc[levelwise["level"] == FINEST_LEVEL].copy() if len(levelwise) else pd.DataFrame()
    edge_df = pd.DataFrame(_parent_child_rows(run_rows))
    subtree_df = pd.DataFrame(_subtree_rows(run_rows))
    study_df = pd.DataFrame(_by_study_rows(run_rows))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    levelwise.to_csv(RESULTS_DIR / "phase3_tradeoff_levelwise.csv", index=False)
    hierarchy.to_csv(RESULTS_DIR / "phase3_tradeoff_hierarchy.csv", index=False)
    edge_df.to_csv(RESULTS_DIR / "phase3_tradeoff_parent_child_breakdown.csv", index=False)
    subtree_df.to_csv(RESULTS_DIR / "phase3_tradeoff_subtree_breakdown.csv", index=False)
    study_df.to_csv(RESULTS_DIR / "phase3_tradeoff_by_study.csv", index=False)
    _write_markdown(finest, hierarchy, edge_df, subtree_df, study_df)
    print(
        json.dumps(
            {
                "predict_rows": len(run_rows),
                "levelwise_rows": len(levelwise),
                "parent_child_rows": len(edge_df),
                "subtree_rows": len(subtree_df),
                "study_rows": len(study_df),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
