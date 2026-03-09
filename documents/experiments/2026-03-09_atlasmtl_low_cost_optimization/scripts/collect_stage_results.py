#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
ROUND_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-09_atlasmtl_low_cost_optimization"
TMP_ROOT = Path("/tmp/atlasmtl_benchmarks/2026-03-09/atlasmtl_low_cost_optimization")
RESULTS_DIR = ROUND_ROOT / "results_summary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["stage_a", "stage_b"], required=True)
    return parser.parse_args()


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_primary_metrics(metrics_payload: Dict[str, Any]) -> Dict[str, Any]:
    result = (metrics_payload.get("results") or [None])[0] or {}
    metrics = result.get("metrics") or {}
    label_columns = result.get("label_columns") or []
    primary_level = label_columns[-1] if label_columns else next(iter(metrics), None)
    primary_metrics = (metrics.get(primary_level) or {}) if primary_level is not None else {}
    train_cfg = result.get("prediction_metadata", {}).get("train_config") or {}
    train_usage = result.get("train_usage") or {}
    predict_usage = result.get("predict_usage") or {}
    final_lr = train_cfg.get("final_learning_rate")
    early_note = "no_validation_split" if train_cfg.get("last_val_loss") is None else (
        "stopped_early" if train_cfg.get("epochs_completed", 0) < train_cfg.get("num_epochs", 0) else "ran_full_epochs"
    )
    if train_cfg.get("scheduler_name") == "reduce_lr_on_plateau" and final_lr is not None:
        if float(final_lr) < float(train_cfg.get("learning_rate", final_lr)):
            early_note = "scheduler_reduced_lr"
        else:
            early_note = "scheduler_without_reduction_visible"
    return {
        "accuracy": primary_metrics.get("accuracy"),
        "macro_f1": primary_metrics.get("macro_f1"),
        "balanced_accuracy": primary_metrics.get("balanced_accuracy"),
        "optimizer_name": train_cfg.get("optimizer_name"),
        "weight_decay": train_cfg.get("weight_decay"),
        "scheduler_name": train_cfg.get("scheduler_name"),
        "epochs_completed": train_cfg.get("epochs_completed"),
        "last_train_loss": train_cfg.get("last_train_loss"),
        "last_val_loss": train_cfg.get("last_val_loss"),
        "train_elapsed_seconds": train_usage.get("elapsed_seconds"),
        "predict_elapsed_seconds": predict_usage.get("elapsed_seconds"),
        "train_process_peak_rss_gb": train_usage.get("process_peak_rss_gb"),
        "predict_process_peak_rss_gb": predict_usage.get("process_peak_rss_gb"),
        "train_gpu_peak_memory_gb": train_usage.get("gpu_peak_memory_gb"),
        "predict_gpu_peak_memory_gb": predict_usage.get("gpu_peak_memory_gb"),
        "device_used": train_usage.get("device_used") or predict_usage.get("device_used"),
        "num_threads_used": train_usage.get("num_threads_used"),
        "early_stopping_note": early_note,
    }


def _scan_stage(stage: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not TMP_ROOT.exists():
        return rows
    for dataset_dir in sorted(p for p in TMP_ROOT.iterdir() if p.is_dir()):
        stage_root = dataset_dir / "benchmark" / stage
        if not stage_root.exists():
            continue
        for track_dir in sorted(p for p in stage_root.iterdir() if p.is_dir()):
            for point_dir in sorted(p for p in track_dir.iterdir() if p.is_dir()):
                for config_dir in sorted(p for p in point_dir.iterdir() if p.is_dir()):
                    status_path = config_dir / "scaleout_status.json"
                    metrics_path = config_dir / "runs" / "atlasmtl" / "metrics.json"
                    stdout_path = config_dir / "runs" / "atlasmtl" / "stdout.log"
                    stderr_path = config_dir / "runs" / "atlasmtl" / "stderr.log"
                    status_payload = _safe_read_json(status_path)
                    metrics_payload = _safe_read_json(metrics_path)
                    row: Dict[str, Any] = {
                        "dataset": dataset_dir.name,
                        "stage": stage,
                        "track": track_dir.name,
                        "point": point_dir.name,
                        "config_name": config_dir.name,
                        "stdout_log": str(stdout_path),
                        "stderr_log": str(stderr_path),
                        "runtime_fairness_degraded": None,
                        "status": "missing_metrics" if metrics_payload is None else "success",
                    }
                    if status_payload is not None:
                        row["runtime_fairness_degraded"] = status_payload.get("runtime_fairness_degraded")
                        methods = status_payload.get("methods") or []
                        atlas_status = next((item for item in methods if item.get("method") == "atlasmtl"), None)
                        if atlas_status is not None:
                            row["status"] = atlas_status.get("status", row["status"])
                    if metrics_payload is not None:
                        row.update(_extract_primary_metrics(metrics_payload))
                    rows.append(row)
    return rows


def _write_outputs(stage: str, rows: List[Dict[str, Any]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / f"{stage}_screening_results.csv"
    md_path = RESULTS_DIR / f"{stage}_screening_results.md"
    if stage == "stage_b":
        csv_path = RESULTS_DIR / "stage_b_confirmation_results.csv"
        md_path = RESULTS_DIR / "stage_b_confirmation_results.md"
    frame.to_csv(csv_path, index=False)

    lines = [f"# {stage.replace('_', ' ').title()} Results", ""]
    lines.append(f"Rows: {len(frame)}")
    lines.append("")
    if not frame.empty:
        cols = [c for c in [
            "dataset",
            "track",
            "point",
            "config_name",
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "train_elapsed_seconds",
            "predict_elapsed_seconds",
            "early_stopping_note",
            "status",
        ] if c in frame.columns]
        lines.append(frame[cols].to_markdown(index=False))
    else:
        lines.append("No runs found.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = _scan_stage(args.stage)
    _write_outputs(args.stage, rows)
    print(json.dumps({"stage": args.stage, "row_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
