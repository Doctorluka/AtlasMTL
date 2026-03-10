#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
DOSSIER_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-10_hlca_study_split_refinement_validation"
RESULTS_DIR = DOSSIER_ROOT / "results_summary"
TMP_ROOT = Path("/tmp/atlasmtl_benchmarks/2026-03-10/hlca_study_split_weight_confirmation")
FINEST_LEVEL = "ann_level_5"


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
        train_config_used = result.get("train_config_used") or {}
        rows[config_dir.name] = {
            "task_weights": json.dumps(train_config_used.get("task_weights"), sort_keys=False),
            "train_elapsed_seconds": runtime["train_elapsed_seconds"],
            "train_process_peak_rss_gb": runtime["train_process_peak_rss_gb"],
            "train_gpu_peak_memory_gb": runtime["train_gpu_peak_memory_gb"],
        }
    return rows


def _scan_predict_rows(train_rows: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    predict_root = TMP_ROOT / "predict"
    if not predict_root.exists():
        return rows
    for config_dir in sorted(p for p in predict_root.iterdir() if p.is_dir()):
        for point_dir in sorted(p for p in config_dir.iterdir() if p.is_dir()):
            run_dir = point_dir / "runs" / "atlasmtl"
            metrics_payload = _safe_read_json(run_dir / "metrics.json")
            if metrics_payload is None:
                continue
            result = _first_result(metrics_payload)
            runtime = _runtime_fields(result)
            rows.append(
                {
                    "dataset": "HLCA_Core",
                    "track": "gpu",
                    "config_name": config_dir.name,
                    "point": point_dir.name,
                    "result": result,
                    **train_rows.get(config_dir.name, {}),
                    "predict_elapsed_seconds": runtime["predict_elapsed_seconds"],
                    "predict_process_peak_rss_gb": runtime["predict_process_peak_rss_gb"],
                    "predict_gpu_peak_memory_gb": runtime["predict_gpu_peak_memory_gb"],
                }
            )
    return rows


def _levelwise_rows(run_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in run_rows:
        metrics = run["result"].get("metrics") or {}
        behavior = run["result"].get("behavior_metrics") or {}
        for level, metric_payload in metrics.items():
            behavior_payload = behavior.get(level) or {}
            rows.append(
                {
                    "dataset": run["dataset"],
                    "track": run["track"],
                    "config_name": run["config_name"],
                    "point": run["point"],
                    "level": level,
                    "accuracy": metric_payload.get("accuracy"),
                    "macro_f1": metric_payload.get("macro_f1"),
                    "balanced_accuracy": metric_payload.get("balanced_accuracy"),
                    "coverage": metric_payload.get("coverage"),
                    "covered_accuracy": metric_payload.get("covered_accuracy"),
                    "unknown_rate": behavior_payload.get("unknown_rate"),
                    "task_weights": run.get("task_weights"),
                    "train_elapsed_seconds": run.get("train_elapsed_seconds"),
                    "predict_elapsed_seconds": run.get("predict_elapsed_seconds"),
                }
            )
    return rows


def _hierarchy_rows(run_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in run_rows:
        hierarchy = run["result"].get("hierarchy_metrics") or {}
        edges = hierarchy.get("edges") or {}
        edge_rates = {child_col: (payload or {}).get("path_consistency_rate") for child_col, payload in edges.items()}
        valid_rates = [float(v) for v in edge_rates.values() if v is not None]
        rows.append(
            {
                "dataset": run["dataset"],
                "track": run["track"],
                "config_name": run["config_name"],
                "point": run["point"],
                "full_path_accuracy": hierarchy.get("full_path_accuracy"),
                "full_path_coverage": hierarchy.get("full_path_coverage"),
                "mean_path_consistency_rate": sum(valid_rates) / len(valid_rates) if valid_rates else None,
            }
        )
    return rows


def _select_best(finest: pd.DataFrame, hierarchy: pd.DataFrame) -> Dict[str, Any]:
    merged = finest.merge(
        hierarchy[["config_name", "point", "full_path_accuracy", "mean_path_consistency_rate"]],
        on=["config_name", "point"],
        how="left",
    )
    target = merged[merged["point"] == "predict_100000_10000"].copy()
    target = target.sort_values(
        by=["macro_f1", "balanced_accuracy", "full_path_accuracy", "config_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    top = target.iloc[0].to_dict()
    mild = target[target["config_name"] == "mild_lv5"]
    strong = target[target["config_name"] == "strong_lv5"]
    chosen = top
    rationale = "highest_macro_f1"
    if not mild.empty and not strong.empty:
        mild_row = mild.iloc[0]
        strong_row = strong.iloc[0]
        if (
            float(strong_row["macro_f1"]) >= float(mild_row["macro_f1"])
            and float(strong_row["macro_f1"]) - float(mild_row["macro_f1"]) <= 0.005
            and float(strong_row["full_path_accuracy"]) <= float(mild_row["full_path_accuracy"]) + 0.005
        ):
            chosen = mild_row.to_dict()
            rationale = "mild_preferred_under_small_macro_gap"
    return {
        "best_config_name": str(chosen["config_name"]),
        "selection_point": "predict_100000_10000",
        "selection_rationale": rationale,
        "macro_f1": float(chosen["macro_f1"]),
        "balanced_accuracy": float(chosen["balanced_accuracy"]),
        "full_path_accuracy": float(chosen["full_path_accuracy"]),
        "coverage": float(chosen["coverage"]),
        "unknown_rate": float(chosen["unknown_rate"]),
    }


def main() -> None:
    train_rows = _load_train_rows()
    run_rows = _scan_predict_rows(train_rows)
    levelwise = pd.DataFrame(_levelwise_rows(run_rows))
    hierarchy = pd.DataFrame(_hierarchy_rows(run_rows))
    finest = levelwise[levelwise["level"] == FINEST_LEVEL].copy() if len(levelwise) else pd.DataFrame()
    best = _select_best(finest, hierarchy) if len(finest) else {}
    comparison = finest.merge(
        hierarchy[["config_name", "point", "full_path_accuracy", "full_path_coverage", "mean_path_consistency_rate"]],
        on=["config_name", "point"],
        how="left",
    ).sort_values(["point", "config_name"]).reset_index(drop=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(RESULTS_DIR / "hlca_weight_confirmation.csv", index=False)
    (RESULTS_DIR / "hlca_weight_confirmation_best_config.json").write_text(
        json.dumps(best, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown = [
        "# HLCA Weight Confirmation",
        "",
        f"- evaluated configs: `{comparison['config_name'].nunique() if len(comparison) else 0}`",
        f"- evaluation rows: `{len(comparison)}`",
        "",
        "## Finest-Level Comparison",
        "",
        comparison[
            [
                "config_name",
                "point",
                "macro_f1",
                "balanced_accuracy",
                "accuracy",
                "coverage",
                "unknown_rate",
                "full_path_accuracy",
            ]
        ].to_markdown(index=False)
        if len(comparison)
        else "No completed runs found.",
        "",
        "## Selected Base Config",
        "",
        json.dumps(best, indent=2, sort_keys=True),
        "",
    ]
    (RESULTS_DIR / "hlca_weight_confirmation.md").write_text("\n".join(markdown), encoding="utf-8")
    print(
        json.dumps(
            {
                "predict_rows": len(run_rows),
                "comparison_rows": len(comparison),
                "best_config_name": best.get("best_config_name"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
