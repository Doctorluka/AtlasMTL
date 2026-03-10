#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
DOSSIER_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-09_phmap_study_split_validation"
RESULTS_DIR = DOSSIER_ROOT / "results_summary"
TMP_ROOT = Path("/tmp/atlasmtl_benchmarks/2026-03-09/phmap_study_split_phase2_screen")
FINEST_LEVEL = "anno_lv4"
BASELINE_CONFIG = "lv4strong_baseline"


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
            "train_elapsed_seconds": runtime["train_elapsed_seconds"],
            "train_process_peak_rss_gb": runtime["train_process_peak_rss_gb"],
            "train_gpu_peak_memory_gb": runtime["train_gpu_peak_memory_gb"],
            "class_weighting": json.dumps(train_config_used.get("class_weighting"), sort_keys=True),
            "class_balanced_sampling": json.dumps(train_config_used.get("class_balanced_sampling"), sort_keys=True),
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
            train_info = train_rows.get(config_dir.name, {})
            rows.append(
                {
                    "dataset": "PHMap_Lung_Full_v43_light",
                    "track": "gpu",
                    "config_name": config_dir.name,
                    "point": point_dir.name,
                    "result": result,
                    "train_elapsed_seconds": train_info.get("train_elapsed_seconds"),
                    "train_process_peak_rss_gb": train_info.get("train_process_peak_rss_gb"),
                    "train_gpu_peak_memory_gb": train_info.get("train_gpu_peak_memory_gb"),
                    "predict_elapsed_seconds": runtime["predict_elapsed_seconds"],
                    "predict_process_peak_rss_gb": runtime["predict_process_peak_rss_gb"],
                    "predict_gpu_peak_memory_gb": runtime["predict_gpu_peak_memory_gb"],
                    "class_weighting": train_info.get("class_weighting"),
                    "class_balanced_sampling": train_info.get("class_balanced_sampling"),
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
                    "level": level,
                    "accuracy": metric_payload.get("accuracy"),
                    "macro_f1": metric_payload.get("macro_f1"),
                    "balanced_accuracy": metric_payload.get("balanced_accuracy"),
                    "coverage": metric_payload.get("coverage"),
                    "covered_accuracy": metric_payload.get("covered_accuracy"),
                    "unknown_rate": behavior_payload.get("unknown_rate"),
                    "train_elapsed_seconds": run["train_elapsed_seconds"],
                    "predict_elapsed_seconds": run["predict_elapsed_seconds"],
                    "class_weighting": run["class_weighting"],
                    "class_balanced_sampling": run["class_balanced_sampling"],
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
                "full_path_accuracy": hierarchy.get("full_path_accuracy"),
                "full_path_coverage": hierarchy.get("full_path_coverage"),
                "full_path_covered_accuracy": hierarchy.get("full_path_covered_accuracy"),
                "mean_path_consistency_rate": sum(valid_rates) / len(valid_rates) if valid_rates else None,
                "min_path_consistency_rate": min(valid_rates) if valid_rates else None,
            }
        )
    return rows


def _choose_best(finest: pd.DataFrame, hierarchy: pd.DataFrame) -> Dict[str, Any]:
    merged = finest.merge(
        hierarchy[["config_name", "point", "full_path_accuracy", "full_path_coverage", "mean_path_consistency_rate"]],
        on=["config_name", "point"],
        how="left",
    )
    target = merged.loc[merged["point"] == "predict_100000_10000"].copy()
    target = target.sort_values(
        by=["macro_f1", "full_path_accuracy", "coverage", "config_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    best = target.iloc[0].to_dict()
    return {
        "best_config_name": best["config_name"],
        "selection_point": "predict_100000_10000",
        "selection_metric": "macro_f1",
        "macro_f1": best["macro_f1"],
        "full_path_accuracy": best.get("full_path_accuracy"),
        "coverage": best.get("coverage"),
    }


def _comparison(finest: pd.DataFrame, hierarchy: pd.DataFrame) -> pd.DataFrame:
    merged = finest.merge(
        hierarchy[["config_name", "point", "full_path_accuracy", "full_path_coverage", "mean_path_consistency_rate"]],
        on=["config_name", "point"],
        how="left",
    )
    baseline = merged.loc[merged["config_name"] == BASELINE_CONFIG].rename(
        columns={
            "macro_f1": "macro_f1_baseline",
            "balanced_accuracy": "balanced_accuracy_baseline",
            "coverage": "coverage_baseline",
            "unknown_rate": "unknown_rate_baseline",
            "full_path_accuracy": "full_path_accuracy_baseline",
        }
    )
    keep = ["point", "macro_f1_baseline", "balanced_accuracy_baseline", "coverage_baseline", "unknown_rate_baseline", "full_path_accuracy_baseline"]
    baseline = baseline[keep]
    out = merged.merge(baseline, on="point", how="left")
    out["delta_macro_f1_vs_baseline"] = out["macro_f1"] - out["macro_f1_baseline"]
    out["delta_full_path_accuracy_vs_baseline"] = out["full_path_accuracy"] - out["full_path_accuracy_baseline"]
    out["delta_coverage_vs_baseline"] = out["coverage"] - out["coverage_baseline"]
    return out.sort_values(["point", "config_name"]).reset_index(drop=True)


def _write_markdown(finest: pd.DataFrame, comparison: pd.DataFrame, best: Dict[str, Any]) -> None:
    lines = [
        "# PH-Map Study-Split Phase 2 Screen",
        "",
        f"- evaluated configs: `{finest['config_name'].nunique() if len(finest) else 0}`",
        f"- evaluation rows: `{len(finest)}`",
        "",
        "## Finest-Level Rows",
        "",
        finest[["config_name", "point", "macro_f1", "balanced_accuracy", "accuracy", "coverage", "unknown_rate", "covered_accuracy"]].to_markdown(index=False) if len(finest) else "No completed runs found.",
        "",
        "## Baseline Comparison",
        "",
        comparison[["config_name", "point", "macro_f1", "delta_macro_f1_vs_baseline", "full_path_accuracy", "delta_full_path_accuracy_vs_baseline", "coverage", "delta_coverage_vs_baseline"]].to_markdown(index=False) if len(comparison) else "No comparison rows found.",
        "",
        "## Selected Candidate",
        "",
        json.dumps(best, indent=2, sort_keys=True),
        "",
    ]
    (RESULTS_DIR / "phase2_screen_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    train_rows = _load_train_rows()
    run_rows = _scan_predict_rows(train_rows)
    levelwise = pd.DataFrame(_levelwise_rows(run_rows))
    hierarchy = pd.DataFrame(_hierarchy_rows(run_rows))
    finest = levelwise.loc[levelwise["level"] == FINEST_LEVEL].copy() if len(levelwise) else pd.DataFrame()
    comparison = _comparison(finest, hierarchy) if len(finest) else pd.DataFrame()
    best = _choose_best(finest, hierarchy) if len(finest) else {}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    levelwise.to_csv(RESULTS_DIR / "phase2_screen_levelwise.csv", index=False)
    hierarchy.to_csv(RESULTS_DIR / "phase2_screen_hierarchy.csv", index=False)
    comparison.to_csv(RESULTS_DIR / "phase2_screen_comparison.csv", index=False)
    (RESULTS_DIR / "phase2_screen_best_config.json").write_text(json.dumps(best, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(finest, comparison, best)
    print(json.dumps({"predict_rows": len(run_rows), "comparison_rows": len(comparison), "best_config_name": best.get("best_config_name")}, indent=2))


if __name__ == "__main__":
    main()
