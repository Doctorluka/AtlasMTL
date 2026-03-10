#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
ROUND_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-09_formal_atlasmtl_refresh"
RESULTS_DIR = ROUND_ROOT / "results_summary"
ANCHOR_PATH = RESULTS_DIR / "atlasmtl_formal_baseline_anchor.csv"
TMP_ROOT = Path("/tmp/atlasmtl_benchmarks/2026-03-09/formal_atlasmtl_refresh")


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_primary(metrics_payload: Dict[str, Any]) -> Dict[str, Any]:
    result = (metrics_payload.get("results") or [None])[0] or {}
    metrics = result.get("metrics") or {}
    label_columns = result.get("label_columns") or []
    primary_level = label_columns[-1] if label_columns else next(iter(metrics), None)
    primary_metrics = (metrics.get(primary_level) or {}) if primary_level is not None else {}
    train_usage = result.get("train_usage") or {}
    predict_usage = result.get("predict_usage") or {}
    return {
        "new_accuracy": primary_metrics.get("accuracy"),
        "new_macro_f1": primary_metrics.get("macro_f1"),
        "new_balanced_accuracy": primary_metrics.get("balanced_accuracy"),
        "new_risk": result.get("behavior_metrics", {}).get(primary_level, {}).get("risk")
        if primary_level is not None
        else None,
        "new_train_elapsed_seconds": train_usage.get("elapsed_seconds"),
        "new_predict_elapsed_seconds": predict_usage.get("elapsed_seconds"),
        "new_train_process_peak_rss_gb": train_usage.get("process_peak_rss_gb"),
        "new_predict_process_peak_rss_gb": predict_usage.get("process_peak_rss_gb"),
        "new_train_gpu_peak_memory_gb": train_usage.get("gpu_peak_memory_gb"),
        "new_predict_gpu_peak_memory_gb": predict_usage.get("gpu_peak_memory_gb"),
        "device_used": train_usage.get("device_used") or predict_usage.get("device_used"),
        "num_threads_used": train_usage.get("num_threads_used"),
    }


def _scan() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not TMP_ROOT.exists():
        return rows
    for dataset_dir in sorted(p for p in TMP_ROOT.iterdir() if p.is_dir()):
        bench_root = dataset_dir / "benchmark"
        if not bench_root.exists():
            continue
        for track_dir in sorted(p for p in bench_root.iterdir() if p.is_dir()):
            for point_dir in sorted(p for p in track_dir.iterdir() if p.is_dir()):
                status_path = point_dir / "scaleout_status.json"
                metrics_path = point_dir / "runs" / "atlasmtl" / "metrics.json"
                stdout_path = point_dir / "runs" / "atlasmtl" / "stdout.log"
                stderr_path = point_dir / "runs" / "atlasmtl" / "stderr.log"
                status_payload = _safe_read_json(status_path)
                metrics_payload = _safe_read_json(metrics_path)
                row: Dict[str, Any] = {
                    "dataset": dataset_dir.name,
                    "track": track_dir.name,
                    "point": point_dir.name,
                    "config_name": "atlasmtl_refreshed_default",
                    "stdout_log": stdout_path.as_posix(),
                    "stderr_log": stderr_path.as_posix(),
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
                    row.update(_extract_primary(metrics_payload))
                rows.append(row)
    return rows


def _write_outputs(rows: List[Dict[str, Any]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    expected_cols = [
        "dataset",
        "track",
        "point",
        "config_name",
        "stdout_log",
        "stderr_log",
        "runtime_fairness_degraded",
        "status",
        "new_accuracy",
        "new_macro_f1",
        "new_balanced_accuracy",
        "new_risk",
        "new_train_elapsed_seconds",
        "new_predict_elapsed_seconds",
        "new_train_process_peak_rss_gb",
        "new_predict_process_peak_rss_gb",
        "new_train_gpu_peak_memory_gb",
        "new_predict_gpu_peak_memory_gb",
        "device_used",
        "num_threads_used",
    ]
    frame = pd.DataFrame(rows, columns=expected_cols)
    anchor = pd.read_csv(ANCHOR_PATH)
    merged = anchor.merge(frame, left_on=["dataset", "track", "point"], right_on=["dataset", "track", "point"], how="left")
    for metric in [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "risk",
        "train_elapsed_seconds",
        "predict_elapsed_seconds",
        "train_process_peak_rss_gb",
        "predict_process_peak_rss_gb",
        "train_gpu_peak_memory_gb",
        "predict_gpu_peak_memory_gb",
    ]:
        merged[f"delta_{metric}"] = merged[f"new_{metric}"] - merged[f"old_{metric}"]

    csv_path = RESULTS_DIR / "formal_refresh_results.csv"
    md_path = RESULTS_DIR / "formal_refresh_results.md"
    merged.to_csv(csv_path, index=False)

    cols = [
        "dataset",
        "track",
        "point",
        "scope",
        "old_macro_f1",
        "new_macro_f1",
        "delta_macro_f1",
        "old_train_elapsed_seconds",
        "new_train_elapsed_seconds",
        "delta_train_elapsed_seconds",
        "status",
    ]
    lines = [
        "# Formal Refresh Results",
        "",
        f"Rows: {len(merged)}",
        "",
        merged[cols].to_markdown(index=False) if len(merged) else "No runs found.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    main = merged[merged["scope"] == "main"].copy()
    gpu = main[main["track"] == "gpu"].copy()
    valid_main = main[main["delta_macro_f1"].notna()].copy()
    valid_gpu = gpu[gpu["delta_macro_f1"].notna()].copy()
    replace = (
        bool(len(valid_main))
        and float(valid_main["delta_macro_f1"].mean()) >= 0.0
        and int((valid_gpu["delta_macro_f1"] > 0).sum()) >= 5
        and float(valid_main["delta_macro_f1"].min()) >= -0.02
        and float(valid_gpu["delta_train_gpu_peak_memory_gb"].median()) <= 0.1
        and float(valid_main["delta_train_process_peak_rss_gb"].median()) <= 0.2
    )
    if not len(valid_main):
        decision_token = "pending_refresh_results"
    else:
        decision_token = "replace_formal_atlasmtl_rows" if replace else "keep_formal_atlasmtl_baseline_rows"
    decision_lines = [
        "# Formal Refresh Decision",
        "",
        f"- main-panel rows: `{len(main)}`",
        f"- main-panel rows with refresh results: `{len(valid_main)}`",
        f"- gpu headline improvements: `{int((valid_gpu['delta_macro_f1'] > 0).sum())}/{len(valid_gpu) if len(valid_gpu) else 0}`",
        f"- main mean delta_macro_f1: `{valid_main['delta_macro_f1'].mean():.6f}`" if len(valid_main) else "- main mean delta_macro_f1: `nan`",
        f"- main min delta_macro_f1: `{valid_main['delta_macro_f1'].min():.6f}`" if len(valid_main) else "- main min delta_macro_f1: `nan`",
        f"- gpu median delta_train_gpu_peak_memory_gb: `{valid_gpu['delta_train_gpu_peak_memory_gb'].median():.6f}`" if len(valid_gpu) else "- gpu median delta_train_gpu_peak_memory_gb: `nan`",
        f"- main median delta_train_process_peak_rss_gb: `{valid_main['delta_train_process_peak_rss_gb'].median():.6f}`" if len(valid_main) else "- main median delta_train_process_peak_rss_gb: `nan`",
        "",
        "Decision:",
        "",
        f"- `{decision_token}`",
        "",
        "Supplementary Vento note:",
        "",
    ]
    vento = merged[merged["dataset"] == "Vento"][["track", "point", "delta_macro_f1"]]
    if len(vento):
        for _, row in vento.iterrows():
            decision_lines.append(f"- `{row['track']} / {row['point']}` delta_macro_f1 = `{row['delta_macro_f1']:.6f}`")
    else:
        decision_lines.append("- no Vento rows found")
    (RESULTS_DIR / "formal_refresh_decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = _scan()
    _write_outputs(rows)
    print(json.dumps({"row_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
