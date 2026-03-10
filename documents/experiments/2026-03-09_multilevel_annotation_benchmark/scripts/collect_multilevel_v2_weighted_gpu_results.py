#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
ROUND_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-09_multilevel_annotation_benchmark"
RESULTS_DIR = ROUND_ROOT / "results_summary" / "v2_weighted_gpu"
TMP_ROOT = Path("/tmp/atlasmtl_benchmarks/2026-03-09/multilevel_annotation_v2_weighted_gpu")
INVENTORY_PATH = ROUND_ROOT / "results_summary" / "dataset_hierarchy_inventory.csv"
V1_LEVELWISE_PATH = ROUND_ROOT / "results_summary" / "levelwise_performance.csv"
V1_HIERARCHY_PATH = ROUND_ROOT / "results_summary" / "hierarchy_performance.csv"
V1_RELIABILITY_PATH = ROUND_ROOT / "results_summary" / "reliability_performance.csv"


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
        bench_root = dataset_dir / "benchmark" / "gpu"
        if not bench_root.exists():
            continue
        for point_dir in sorted(p for p in bench_root.iterdir() if p.is_dir()):
            metrics_payload = _safe_read_json(point_dir / "runs" / "atlasmtl" / "metrics.json")
            row = {
                "dataset": dataset_dir.name,
                "track": "gpu",
                "point": point_dir.name,
                "status": "missing_metrics" if metrics_payload is None else "success",
                "runtime_fairness_degraded": False,
                "stdout_log": (point_dir / "runs" / "atlasmtl" / "stdout.log").as_posix(),
                "stderr_log": (point_dir / "runs" / "atlasmtl" / "stderr.log").as_posix(),
            }
            if metrics_payload is not None:
                row["result"] = _first_result(metrics_payload)
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
                    "runtime_fairness_degraded": False,
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
        edge_rates = {child_col: (payload or {}).get("path_consistency_rate") for child_col, payload in edges.items()}
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
                "runtime_fairness_degraded": False,
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
                "runtime_fairness_degraded": False,
                "status": run["status"],
            }
        )
    return rows


def _comparison(levelwise: pd.DataFrame, hierarchy: pd.DataFrame, reliability: pd.DataFrame) -> pd.DataFrame:
    v1_levelwise = pd.read_csv(V1_LEVELWISE_PATH)
    v1_hierarchy = pd.read_csv(V1_HIERARCHY_PATH)
    v1_reliability = pd.read_csv(V1_RELIABILITY_PATH)
    v1_finest = v1_levelwise.merge(
        v1_reliability[["dataset", "track", "point", "finest_level"]],
        left_on=["dataset", "track", "point", "level"],
        right_on=["dataset", "track", "point", "finest_level"],
        how="inner",
    )
    v2_finest = levelwise.merge(
        reliability[["dataset", "track", "point", "finest_level"]],
        left_on=["dataset", "track", "point", "level"],
        right_on=["dataset", "track", "point", "finest_level"],
        how="inner",
    )
    keep_v = [
        "dataset",
        "track",
        "point",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "coverage",
        "covered_accuracy",
        "risk",
        "train_elapsed_seconds",
        "train_gpu_peak_memory_gb",
    ]
    keep_h = ["dataset", "track", "point", "full_path_accuracy", "full_path_coverage", "full_path_covered_accuracy"]
    v1_hierarchy = v1_hierarchy[keep_h].rename(
        columns={
            "full_path_accuracy": "full_path_accuracy_v1",
            "full_path_coverage": "full_path_coverage_v1",
            "full_path_covered_accuracy": "full_path_covered_accuracy_v1",
        }
    )
    merged = (
        v1_finest[keep_v]
        .merge(v1_hierarchy, on=["dataset", "track", "point"], how="left")
        .merge(v2_finest[keep_v], on=["dataset", "track", "point"], how="inner", suffixes=("_v1", "_v2"))
        .merge(hierarchy[keep_h], on=["dataset", "track", "point"], how="left")
    )
    rename_map = {
        "full_path_accuracy": "full_path_accuracy_v2",
        "full_path_coverage": "full_path_coverage_v2",
        "full_path_covered_accuracy": "full_path_covered_accuracy_v2",
    }
    merged = merged.rename(columns=rename_map)
    for metric in [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "coverage",
        "covered_accuracy",
        "risk",
        "train_elapsed_seconds",
        "train_gpu_peak_memory_gb",
        "full_path_accuracy",
        "full_path_coverage",
        "full_path_covered_accuracy",
    ]:
        merged[f"delta_{metric}"] = merged[f"{metric}_v2"] - merged[f"{metric}_v1"]
    return merged.sort_values(["dataset", "point"]).reset_index(drop=True)


def _write_reports(levelwise: pd.DataFrame, hierarchy: pd.DataFrame, reliability: pd.DataFrame, comparison: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    levelwise.to_csv(RESULTS_DIR / "levelwise_performance.csv", index=False)
    hierarchy.to_csv(RESULTS_DIR / "hierarchy_performance.csv", index=False)
    reliability.to_csv(RESULTS_DIR / "reliability_performance.csv", index=False)
    comparison.to_csv(RESULTS_DIR / "comparison_vs_v1_uniform_gpu.csv", index=False)

    md_lines = [
        "# V2 Weighted GPU Vs V1 Uniform GPU",
        "",
        f"Rows: {len(comparison)}",
        "",
        comparison[
            [
                "dataset",
                "point",
                "macro_f1_v1",
                "macro_f1_v2",
                "delta_macro_f1",
                "full_path_accuracy_v1",
                "full_path_accuracy_v2",
                "delta_full_path_accuracy",
                "train_elapsed_seconds_v1",
                "train_elapsed_seconds_v2",
            ]
        ].to_markdown(index=False)
        if len(comparison)
        else "No completed v2 runs found.",
        "",
    ]
    (RESULTS_DIR / "comparison_vs_v1_uniform_gpu.md").write_text("\n".join(md_lines), encoding="utf-8")

    if len(comparison):
        by_dataset = comparison.groupby("dataset", as_index=False).agg(
            mean_delta_macro_f1=("delta_macro_f1", "mean"),
            mean_delta_full_path_accuracy=("delta_full_path_accuracy", "mean"),
            mean_delta_accuracy=("delta_accuracy", "mean"),
            mean_delta_coverage=("delta_coverage", "mean"),
        )
        positive = int((by_dataset["mean_delta_macro_f1"] >= 0).sum())
        phmap_mean = float(
            by_dataset.loc[by_dataset["dataset"] == "PHMap_Lung_Full_v43_light", "mean_delta_macro_f1"].iloc[0]
        )
        max_full_path_drop = float(by_dataset["mean_delta_full_path_accuracy"].min())
        promote = positive >= 3 and phmap_mean > 0.0 and max_full_path_drop >= -0.01
    else:
        by_dataset = pd.DataFrame()
        positive = 0
        phmap_mean = float("nan")
        max_full_path_drop = float("nan")
        promote = False

    report_lines = [
        "# V2 Weighted GPU Report",
        "",
        f"- completed gpu rows: `{len(comparison)}`",
        f"- datasets with non-negative mean delta_macro_f1: `{positive}/4`" if len(comparison) else "- datasets with non-negative mean delta_macro_f1: `0/4`",
        f"- PHMap mean delta_macro_f1: `{phmap_mean:.6f}`" if len(comparison) else "- PHMap mean delta_macro_f1: `nan`",
        f"- minimum dataset mean delta_full_path_accuracy: `{max_full_path_drop:.6f}`" if len(comparison) else "- minimum dataset mean delta_full_path_accuracy: `nan`",
        "",
    ]
    if len(by_dataset):
        report_lines.extend(["## Dataset Means", "", by_dataset.to_markdown(index=False), ""])
    (RESULTS_DIR / "v2_weighted_gpu_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    decision_lines = [
        "# V2 Weighted GPU Decision Note",
        "",
        "Decision rule:",
        "",
        "- at least `3/4` datasets have non-negative mean `delta_macro_f1`",
        "- `PHMap_Lung_Full_v43_light` mean `delta_macro_f1` is positive",
        "- no dataset mean `delta_full_path_accuracy` falls below `-0.01`",
        "",
        f"Current decision: `{'promote_v2_over_v1' if promote else 'keep_v1_as_primary_sixth_round_track'}`",
        "",
        "v1 remains frozen regardless of this outcome.",
        "",
    ]
    if len(by_dataset):
        decision_lines.extend([by_dataset.to_markdown(index=False), ""])
    (RESULTS_DIR / "v2_weighted_gpu_decision_note.md").write_text("\n".join(decision_lines), encoding="utf-8")


def main() -> None:
    inventory = pd.read_csv(INVENTORY_PATH)
    run_rows = _scan_runs()
    levelwise = pd.DataFrame(_levelwise_rows(run_rows, inventory))
    hierarchy = pd.DataFrame(_hierarchy_rows(run_rows))
    reliability = pd.DataFrame(_reliability_rows(run_rows, inventory))
    comparison = _comparison(levelwise, hierarchy, reliability) if len(levelwise) else pd.DataFrame()
    _write_reports(levelwise, hierarchy, reliability, comparison)
    print(
        json.dumps(
            {
                "run_count": len(run_rows),
                "levelwise_rows": len(levelwise),
                "hierarchy_rows": len(hierarchy),
                "reliability_rows": len(reliability),
                "comparison_rows": len(comparison),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
