#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
DOSSIER_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-09_phmap_study_split_validation"
RESULTS_DIR = DOSSIER_ROOT / "results_summary"
TMP_ROOT = Path("/tmp/atlasmtl_benchmarks/2026-03-09/phmap_study_split_phase2_seed")
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
    }


def _load_train_rows() -> Dict[str, Dict[int, Dict[str, Any]]]:
    rows: Dict[str, Dict[int, Dict[str, Any]]] = {}
    train_root = TMP_ROOT / "train"
    if not train_root.exists():
        return rows
    for config_dir in sorted(p for p in train_root.iterdir() if p.is_dir()):
        rows[config_dir.name] = {}
        for seed_dir in sorted(p for p in config_dir.iterdir() if p.is_dir()):
            seed = int(seed_dir.name.replace("seed_", ""))
            run_dir = seed_dir / "runs" / "atlasmtl"
            metrics_payload = _safe_read_json(run_dir / "metrics.json")
            if metrics_payload is None:
                continue
            result = _first_result(metrics_payload)
            runtime = _runtime_fields(result)
            rows[config_dir.name][seed] = {
                "train_elapsed_seconds": runtime["train_elapsed_seconds"],
            }
    return rows


def _scan_predict_rows(train_rows: Dict[str, Dict[int, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    predict_root = TMP_ROOT / "predict"
    if not predict_root.exists():
        return rows
    for config_dir in sorted(p for p in predict_root.iterdir() if p.is_dir()):
        for seed_dir in sorted(p for p in config_dir.iterdir() if p.is_dir()):
            seed = int(seed_dir.name.replace("seed_", ""))
            for point_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
                run_dir = point_dir / "runs" / "atlasmtl"
                metrics_payload = _safe_read_json(run_dir / "metrics.json")
                if metrics_payload is None:
                    continue
                result = _first_result(metrics_payload)
                runtime = _runtime_fields(result)
                train_info = (train_rows.get(config_dir.name) or {}).get(seed, {})
                rows.append(
                    {
                        "config_name": config_dir.name,
                        "seed": seed,
                        "point": point_dir.name,
                        "result": result,
                        "train_elapsed_seconds": train_info.get("train_elapsed_seconds"),
                        "predict_elapsed_seconds": runtime["predict_elapsed_seconds"],
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
                    "config_name": run["config_name"],
                    "seed": run["seed"],
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
                }
            )
    return rows


def _hierarchy_rows(run_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in run_rows:
        hierarchy = (run["result"].get("hierarchy_metrics") or {})
        rows.append(
            {
                "config_name": run["config_name"],
                "seed": run["seed"],
                "point": run["point"],
                "full_path_accuracy": hierarchy.get("full_path_accuracy"),
                "full_path_coverage": hierarchy.get("full_path_coverage"),
                "full_path_covered_accuracy": hierarchy.get("full_path_covered_accuracy"),
            }
        )
    return rows


def _aggregate(finest: pd.DataFrame, hierarchy: pd.DataFrame) -> pd.DataFrame:
    merged = finest.merge(hierarchy, on=["config_name", "seed", "point"], how="left")
    grouped = (
        merged.groupby(["config_name", "point"], as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            full_path_accuracy_mean=("full_path_accuracy", "mean"),
            full_path_accuracy_std=("full_path_accuracy", "std"),
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            unknown_rate_mean=("unknown_rate", "mean"),
            unknown_rate_std=("unknown_rate", "std"),
        )
    )
    return grouped.sort_values(["point", "config_name"]).reset_index(drop=True)


def _write_markdown(aggregate: pd.DataFrame) -> None:
    lines = [
        "# PH-Map Study-Split Phase 2 Seed Stability",
        "",
        aggregate.to_markdown(index=False) if len(aggregate) else "No completed runs found.",
        "",
    ]
    (RESULTS_DIR / "phase2_seed_stability.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    train_rows = _load_train_rows()
    run_rows = _scan_predict_rows(train_rows)
    levelwise = pd.DataFrame(_levelwise_rows(run_rows))
    hierarchy = pd.DataFrame(_hierarchy_rows(run_rows))
    finest = levelwise.loc[levelwise["level"] == FINEST_LEVEL].copy() if len(levelwise) else pd.DataFrame()
    aggregate = _aggregate(finest, hierarchy) if len(finest) else pd.DataFrame()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    levelwise.to_csv(RESULTS_DIR / "phase2_seed_levelwise.csv", index=False)
    hierarchy.to_csv(RESULTS_DIR / "phase2_seed_hierarchy.csv", index=False)
    aggregate.to_csv(RESULTS_DIR / "phase2_seed_summary.csv", index=False)
    _write_markdown(aggregate)
    print(json.dumps({"predict_rows": len(run_rows), "aggregate_rows": len(aggregate)}, indent=2))


if __name__ == "__main__":
    main()
