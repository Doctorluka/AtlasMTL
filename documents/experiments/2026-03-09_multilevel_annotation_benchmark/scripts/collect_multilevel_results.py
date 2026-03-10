#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
ROUND_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-09_multilevel_annotation_benchmark"
RESULTS_DIR = ROUND_ROOT / "results_summary"
TMP_ROOT = Path("/tmp/atlasmtl_benchmarks/2026-03-09/multilevel_annotation")
INVENTORY_PATH = RESULTS_DIR / "dataset_hierarchy_inventory.csv"


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
        "device_used": train_usage.get("device_used") or predict_usage.get("device_used"),
        "num_threads_used": train_usage.get("num_threads_used"),
    }


def _scan_runs() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not TMP_ROOT.exists():
        return rows
    for dataset_dir in sorted(p for p in TMP_ROOT.iterdir() if p.is_dir()):
        bench_root = dataset_dir / "benchmark"
        if not bench_root.exists():
            continue
        for track_dir in sorted(p for p in bench_root.iterdir() if p.is_dir()):
            for point_dir in sorted(p for p in track_dir.iterdir() if p.is_dir()):
                status_payload = _safe_read_json(point_dir / "scaleout_status.json")
                metrics_payload = _safe_read_json(point_dir / "runs" / "atlasmtl" / "metrics.json")
                row = {
                    "dataset": dataset_dir.name,
                    "track": track_dir.name,
                    "point": point_dir.name,
                    "status": "missing_metrics" if metrics_payload is None else "success",
                    "runtime_fairness_degraded": None,
                    "stdout_log": (point_dir / "runs" / "atlasmtl" / "stdout.log").as_posix(),
                    "stderr_log": (point_dir / "runs" / "atlasmtl" / "stderr.log").as_posix(),
                }
                if status_payload is not None:
                    row["runtime_fairness_degraded"] = status_payload.get("runtime_fairness_degraded")
                    atlas = next((item for item in (status_payload.get("methods") or []) if item.get("method") == "atlasmtl"), None)
                    if atlas is not None:
                        row["status"] = atlas.get("status", row["status"])
                if metrics_payload is not None:
                    result = _first_result(metrics_payload)
                    row["result"] = result
                rows.append(row)
    return rows


def _levelwise_rows(run_rows: List[Dict[str, Any]], inventory: pd.DataFrame) -> List[Dict[str, Any]]:
    class_counts = {
        row["dataset"]: dict(zip(json.loads(row["level_columns"]), json.loads(row["n_classes_by_level"])))
        for _, row in inventory.iterrows()
    }
    rows: List[Dict[str, Any]] = []
    for run in run_rows:
        result = run.get("result")
        if not result:
            continue
        metrics = result.get("metrics") or {}
        behavior = result.get("behavior_metrics") or {}
        runtime = _runtime_fields(result)
        for level, metric_payload in metrics.items():
            behavior_payload = behavior.get(level) or {}
            rows.append(
                {
                    "dataset": run["dataset"],
                    "track": run["track"],
                    "point": run["point"],
                    "level": level,
                    "n_classes": class_counts.get(run["dataset"], {}).get(level),
                    "accuracy": metric_payload.get("accuracy"),
                    "macro_f1": metric_payload.get("macro_f1"),
                    "balanced_accuracy": metric_payload.get("balanced_accuracy"),
                    "coverage": metric_payload.get("coverage"),
                    "reject_rate": metric_payload.get("reject_rate"),
                    "covered_accuracy": metric_payload.get("covered_accuracy"),
                    "risk": metric_payload.get("risk"),
                    "ece": metric_payload.get("ece"),
                    "brier": metric_payload.get("brier"),
                    "aurc": metric_payload.get("aurc"),
                    "unknown_rate": behavior_payload.get("unknown_rate"),
                    "train_elapsed_seconds": runtime["train_elapsed_seconds"],
                    "predict_elapsed_seconds": runtime["predict_elapsed_seconds"],
                    "train_process_peak_rss_gb": runtime["train_process_peak_rss_gb"],
                    "predict_process_peak_rss_gb": runtime["predict_process_peak_rss_gb"],
                    "train_gpu_peak_memory_gb": runtime["train_gpu_peak_memory_gb"],
                    "predict_gpu_peak_memory_gb": runtime["predict_gpu_peak_memory_gb"],
                    "runtime_fairness_degraded": run["runtime_fairness_degraded"],
                    "status": run["status"],
                    "stdout_log": run["stdout_log"],
                    "stderr_log": run["stderr_log"],
                }
            )
    return rows


def _hierarchy_rows(run_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in run_rows:
        result = run.get("result")
        if not result:
            continue
        hierarchy = result.get("hierarchy_metrics") or {}
        edges = hierarchy.get("edges") or {}
        edge_rates = {
            child_col: (payload or {}).get("path_consistency_rate")
            for child_col, payload in edges.items()
        }
        valid_rates = [float(v) for v in edge_rates.values() if v is not None]
        rows.append(
            {
                "dataset": run["dataset"],
                "track": run["track"],
                "point": run["point"],
                "full_path_accuracy": hierarchy.get("full_path_accuracy"),
                "full_path_coverage": hierarchy.get("full_path_coverage"),
                "full_path_covered_accuracy": hierarchy.get("full_path_covered_accuracy"),
                "mean_path_consistency_rate": sum(valid_rates) / len(valid_rates) if valid_rates else None,
                "min_path_consistency_rate": min(valid_rates) if valid_rates else None,
                "path_edge_count": len(edge_rates),
                "hierarchy_enforced": True,
                "edge_path_consistency_json": json.dumps(edge_rates, sort_keys=True),
                "runtime_fairness_degraded": run["runtime_fairness_degraded"],
                "status": run["status"],
            }
        )
    return rows


def _reliability_rows(run_rows: List[Dict[str, Any]], inventory: pd.DataFrame) -> List[Dict[str, Any]]:
    finest_map = {row["dataset"]: row["finest_label_column"] for _, row in inventory.iterrows()}
    rows: List[Dict[str, Any]] = []
    for run in run_rows:
        result = run.get("result")
        if not result:
            continue
        finest = finest_map.get(run["dataset"])
        metrics = (result.get("metrics") or {}).get(finest) or {}
        behavior = (result.get("behavior_metrics") or {}).get(finest) or {}
        rows.append(
            {
                "dataset": run["dataset"],
                "track": run["track"],
                "point": run["point"],
                "finest_level": finest,
                "coverage": metrics.get("coverage"),
                "reject_rate": metrics.get("reject_rate"),
                "covered_accuracy": metrics.get("covered_accuracy"),
                "risk": metrics.get("risk"),
                "ece": metrics.get("ece"),
                "brier": metrics.get("brier"),
                "aurc": metrics.get("aurc"),
                "unknown_rate": behavior.get("unknown_rate"),
                "runtime_fairness_degraded": run["runtime_fairness_degraded"],
                "status": run["status"],
            }
        )
    return rows


def _write_markdown(levelwise: pd.DataFrame, hierarchy: pd.DataFrame, reliability: pd.DataFrame) -> None:
    summary_lines = [
        "# Multi-Level Benchmark Summary",
        "",
        f"- level-wise rows: `{len(levelwise)}`",
        f"- hierarchy rows: `{len(hierarchy)}`",
        f"- reliability rows: `{len(reliability)}`",
        "",
    ]
    if len(levelwise):
        macro = (
            levelwise.groupby(["dataset", "track", "level"], as_index=False)["macro_f1"]
            .mean()
            .sort_values(["dataset", "track", "level"])
        )
        summary_lines.extend(["## Mean Macro-F1 By Level", "", macro.to_markdown(index=False), ""])
    else:
        summary_lines.extend(["No completed runs found.", ""])
    if len(hierarchy):
        cols = [
            "dataset",
            "track",
            "point",
            "full_path_accuracy",
            "full_path_coverage",
            "full_path_covered_accuracy",
            "mean_path_consistency_rate",
            "min_path_consistency_rate",
        ]
        summary_lines.extend(["## Hierarchy Summary", "", hierarchy[cols].to_markdown(index=False), ""])
    if len(reliability):
        cols = [
            "dataset",
            "track",
            "point",
            "finest_level",
            "coverage",
            "reject_rate",
            "covered_accuracy",
            "risk",
            "ece",
            "aurc",
        ]
        summary_lines.extend(["## Finest-Level Reliability", "", reliability[cols].to_markdown(index=False), ""])
    (RESULTS_DIR / "multilevel_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    decision_lines = [
        "# Multi-Level Decision Note",
        "",
        "This round is intended to validate AtlasMTL's multi-level annotation claim.",
        "",
        "Interpretation reminders:",
        "",
        "- KNN is off for all runs in this round.",
        "- CPU degraded-runtime flags should be treated as caveats rather than method failures.",
        "- Main analysis should compare coarse-to-fine performance, full-path quality, and finest-level reliability.",
        "",
    ]
    if len(hierarchy):
        best = hierarchy.sort_values(["full_path_accuracy", "mean_path_consistency_rate"], ascending=False).head(8)
        decision_lines.extend(["Top hierarchy rows observed so far:", "", best.to_markdown(index=False), ""])
    else:
        decision_lines.extend(["No hierarchy results have been collected yet.", ""])
    (RESULTS_DIR / "multilevel_decision_note.md").write_text("\n".join(decision_lines), encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    inventory = pd.read_csv(INVENTORY_PATH) if INVENTORY_PATH.exists() else pd.DataFrame()
    run_rows = _scan_runs()
    levelwise = pd.DataFrame(_levelwise_rows(run_rows, inventory)) if len(inventory) else pd.DataFrame()
    hierarchy = pd.DataFrame(_hierarchy_rows(run_rows))
    reliability = pd.DataFrame(_reliability_rows(run_rows, inventory)) if len(inventory) else pd.DataFrame()

    levelwise.to_csv(RESULTS_DIR / "levelwise_performance.csv", index=False)
    hierarchy.to_csv(RESULTS_DIR / "hierarchy_performance.csv", index=False)
    reliability.to_csv(RESULTS_DIR / "reliability_performance.csv", index=False)
    _write_markdown(levelwise, hierarchy, reliability)
    print(
        json.dumps(
            {
                "run_count": len(run_rows),
                "levelwise_rows": len(levelwise),
                "hierarchy_rows": len(hierarchy),
                "reliability_rows": len(reliability),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
