#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
DOSSIER_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-09_phmap_study_split_validation"
RESULTS_DIR = DOSSIER_ROOT / "results_summary"
TMP_ROOT = Path("/tmp/atlasmtl_benchmarks/2026-03-09/phmap_study_split_phase1")
FINEST_LEVEL = "anno_lv4"


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
        runtime = _runtime_fields(_first_result(metrics_payload))
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
                rows.append(
                    {
                        "dataset": "PHMap_Lung_Full_v43_light",
                        "track": "gpu",
                        "config_name": config_dir.name,
                        "point": point_dir.name,
                        "hierarchy_setting": hierarchy_dir.name.replace("hierarchy_", ""),
                        "enforce_hierarchy": hierarchy_dir.name.endswith("_on"),
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
                    "reject_rate": metric_payload.get("reject_rate"),
                    "covered_accuracy": metric_payload.get("covered_accuracy"),
                    "risk": metric_payload.get("risk"),
                    "ece": metric_payload.get("ece"),
                    "brier": metric_payload.get("brier"),
                    "aurc": metric_payload.get("aurc"),
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


def _reliability_rows(levelwise: pd.DataFrame) -> pd.DataFrame:
    return levelwise.loc[levelwise["level"] == FINEST_LEVEL].copy()


def _weight_comparison(finest: pd.DataFrame, hierarchy: pd.DataFrame) -> pd.DataFrame:
    merge_keys = ["dataset", "track", "point", "hierarchy_setting", "enforce_hierarchy"]
    keep_cols = ["accuracy", "macro_f1", "balanced_accuracy", "coverage", "covered_accuracy", "risk", "unknown_rate", "train_elapsed_seconds", "predict_elapsed_seconds"]
    uniform = finest.loc[finest["config_name"] == "uniform_control", merge_keys + keep_cols]
    lv4 = finest.loc[finest["config_name"] == "lv4strong_candidate", merge_keys + keep_cols]
    merged = uniform.merge(lv4, on=merge_keys, suffixes=("_uniform", "_lv4strong"))
    hu = hierarchy.loc[hierarchy["config_name"] == "uniform_control", merge_keys + ["full_path_accuracy", "full_path_coverage", "full_path_covered_accuracy", "mean_path_consistency_rate", "min_path_consistency_rate"]].rename(columns=lambda c: c if c in merge_keys else f"{c}_uniform")
    hl = hierarchy.loc[hierarchy["config_name"] == "lv4strong_candidate", merge_keys + ["full_path_accuracy", "full_path_coverage", "full_path_covered_accuracy", "mean_path_consistency_rate", "min_path_consistency_rate"]].rename(columns=lambda c: c if c in merge_keys else f"{c}_lv4strong")
    merged = merged.merge(hu, on=merge_keys, how="left").merge(hl, on=merge_keys, how="left")
    for metric in ["accuracy", "macro_f1", "balanced_accuracy", "coverage", "covered_accuracy", "risk", "unknown_rate", "predict_elapsed_seconds", "full_path_accuracy", "full_path_coverage", "full_path_covered_accuracy", "mean_path_consistency_rate", "min_path_consistency_rate"]:
        merged[f"delta_{metric}"] = merged[f"{metric}_lv4strong"] - merged[f"{metric}_uniform"]
    return merged.sort_values(["point", "hierarchy_setting"]).reset_index(drop=True)


def _hierarchy_delta(finest: pd.DataFrame, hierarchy: pd.DataFrame) -> pd.DataFrame:
    merge_keys = ["dataset", "track", "config_name", "point"]
    keep_cols = ["accuracy", "macro_f1", "balanced_accuracy", "coverage", "covered_accuracy", "risk", "unknown_rate", "predict_elapsed_seconds"]
    on_rows = finest.loc[finest["hierarchy_setting"] == "on", merge_keys + keep_cols].rename(columns={c: f"{c}_on" for c in keep_cols})
    off_rows = finest.loc[finest["hierarchy_setting"] == "off", merge_keys + keep_cols].rename(columns={c: f"{c}_off" for c in keep_cols})
    merged = on_rows.merge(off_rows, on=merge_keys, how="inner")
    ho = hierarchy.loc[hierarchy["hierarchy_setting"] == "on", merge_keys + ["full_path_accuracy", "full_path_coverage", "full_path_covered_accuracy", "mean_path_consistency_rate", "min_path_consistency_rate"]].rename(columns=lambda c: c if c in merge_keys else f"{c}_on")
    hf = hierarchy.loc[hierarchy["hierarchy_setting"] == "off", merge_keys + ["full_path_accuracy", "full_path_coverage", "full_path_covered_accuracy", "mean_path_consistency_rate", "min_path_consistency_rate"]].rename(columns=lambda c: c if c in merge_keys else f"{c}_off")
    merged = merged.merge(ho, on=merge_keys, how="left").merge(hf, on=merge_keys, how="left")
    for metric in ["accuracy", "macro_f1", "balanced_accuracy", "coverage", "covered_accuracy", "risk", "unknown_rate", "predict_elapsed_seconds", "full_path_accuracy", "full_path_coverage", "full_path_covered_accuracy", "mean_path_consistency_rate", "min_path_consistency_rate"]:
        merged[f"delta_off_minus_on_{metric}"] = merged[f"{metric}_off"] - merged[f"{metric}_on"]
    return merged.sort_values(["config_name", "point"]).reset_index(drop=True)


def _write_markdown(finest: pd.DataFrame, comparison: pd.DataFrame, hierarchy_delta: pd.DataFrame) -> None:
    lines = [
        "# PH-Map Study-Split Phase 1",
        "",
        f"- evaluation rows: `{len(finest)}`",
        "",
        "## Finest-Level Rows",
        "",
        finest[["config_name", "point", "hierarchy_setting", "macro_f1", "balanced_accuracy", "accuracy", "coverage", "unknown_rate", "covered_accuracy"]].to_markdown(index=False) if len(finest) else "No completed runs found.",
        "",
        "## Weight Comparison",
        "",
        comparison[["point", "hierarchy_setting", "macro_f1_uniform", "macro_f1_lv4strong", "delta_macro_f1", "full_path_accuracy_uniform", "full_path_accuracy_lv4strong", "delta_full_path_accuracy", "coverage_uniform", "coverage_lv4strong", "delta_coverage"]].to_markdown(index=False) if len(comparison) else "No comparison rows found.",
        "",
        "## Hierarchy Tradeoff",
        "",
        hierarchy_delta[["config_name", "point", "macro_f1_on", "macro_f1_off", "delta_off_minus_on_macro_f1", "coverage_on", "coverage_off", "delta_off_minus_on_coverage", "mean_path_consistency_rate_on", "mean_path_consistency_rate_off"]].to_markdown(index=False) if len(hierarchy_delta) else "No hierarchy delta rows found.",
        "",
    ]
    (RESULTS_DIR / "phase1_weight_and_hierarchy_ablation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    train_rows = _load_train_rows()
    run_rows = _scan_predict_rows(train_rows)
    levelwise = pd.DataFrame(_levelwise_rows(run_rows))
    hierarchy = pd.DataFrame(_hierarchy_rows(run_rows))
    reliability = _reliability_rows(levelwise) if len(levelwise) else pd.DataFrame()
    finest = levelwise.loc[levelwise["level"] == FINEST_LEVEL].copy() if len(levelwise) else pd.DataFrame()
    comparison = _weight_comparison(finest, hierarchy) if len(finest) else pd.DataFrame()
    hierarchy_delta = _hierarchy_delta(finest, hierarchy) if len(finest) else pd.DataFrame()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    levelwise.to_csv(RESULTS_DIR / "phase1_levelwise.csv", index=False)
    hierarchy.to_csv(RESULTS_DIR / "phase1_hierarchy.csv", index=False)
    reliability.to_csv(RESULTS_DIR / "phase1_reliability.csv", index=False)
    comparison.to_csv(RESULTS_DIR / "phase1_comparison.csv", index=False)
    hierarchy_delta.to_csv(RESULTS_DIR / "phase1_hierarchy_delta.csv", index=False)
    _write_markdown(finest, comparison, hierarchy_delta)
    print(json.dumps({"predict_rows": len(run_rows), "levelwise_rows": len(levelwise), "comparison_rows": len(comparison)}, indent=2))


if __name__ == "__main__":
    main()
