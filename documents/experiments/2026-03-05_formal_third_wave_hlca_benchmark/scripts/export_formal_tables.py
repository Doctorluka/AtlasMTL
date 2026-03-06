#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpu-runs-dir",
        default="/tmp/atlasmtl_benchmarks/2026-03-05/reference_heldout/HLCA_Core/benchmark/formal_train10k_test5k/cpu_group_v1/runs",
    )
    parser.add_argument(
        "--gpu-runs-dir",
        default="/tmp/atlasmtl_benchmarks/2026-03-05/reference_heldout/HLCA_Core/benchmark/formal_train10k_test5k/gpu_group_v1/runs",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/data/fhz/project/phmap_package/atlasmtl/documents/experiments/2026-03-05_formal_third_wave_hlca_benchmark/results_summary",
    )
    return parser.parse_args()


def _load_result(metrics_json: Path) -> Dict[str, Any]:
    payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    results = list(payload.get("results") or [])
    if not results:
        raise ValueError(f"empty results in {metrics_json}")
    return dict(results[0])


def _collect_rows(group: str, runs_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    perf_rows: List[Dict[str, Any]] = []
    res_rows: List[Dict[str, Any]] = []
    for metrics_json in sorted(runs_dir.glob("*/metrics.json")):
        result = _load_result(metrics_json)
        method = str(result.get("method"))
        level_metrics = next(iter((result.get("metrics") or {}).values()), {})
        behavior_metrics = next(iter((result.get("behavior_metrics") or {}).values()), {})
        fairness = dict(result.get("fairness_metadata") or {})
        train = dict(result.get("train_usage") or {})
        pred = dict(result.get("predict_usage") or {})
        perf_rows.append(
            {
                "group": group,
                "method": method,
                "accuracy": level_metrics.get("accuracy"),
                "macro_f1": level_metrics.get("macro_f1"),
                "ece": level_metrics.get("ece"),
                "aurc": level_metrics.get("aurc"),
                "reject_rate": behavior_metrics.get("reject_rate", level_metrics.get("reject_rate")),
            }
        )
        res_rows.append(
            {
                "group": group,
                "method": method,
                "device_used": pred.get("device_used") or train.get("device_used"),
                "method_backend_path": fairness.get("method_backend_path"),
                "runtime_fairness_degraded": fairness.get("runtime_fairness_degraded"),
                "train_elapsed_seconds": train.get("elapsed_seconds"),
                "predict_elapsed_seconds": pred.get("elapsed_seconds"),
                "train_process_peak_rss_gb": train.get("process_peak_rss_gb"),
                "predict_process_peak_rss_gb": pred.get("process_peak_rss_gb"),
                "train_gpu_peak_memory_gb": train.get("gpu_peak_memory_gb"),
                "predict_gpu_peak_memory_gb": pred.get("gpu_peak_memory_gb"),
                "train_items_per_second": train.get("items_per_second"),
                "predict_items_per_second": pred.get("items_per_second"),
                "effective_threads_observed": fairness.get("effective_threads_observed"),
            }
        )
    return perf_rows, res_rows


def _write_markdown_table(path: Path, title: str, frame: pd.DataFrame) -> None:
    lines = [f"# {title}", ""]
    if frame.empty:
        lines.append("_no rows_")
    else:
        try:
            lines.append(frame.to_markdown(index=False))
        except Exception:
            columns = list(frame.columns)
            lines.append("| " + " | ".join(columns) + " |")
            lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
            for row in frame.itertuples(index=False, name=None):
                values = ["" if value is None else str(value) for value in row]
                lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cpu_runs = Path(args.cpu_runs_dir).resolve()
    gpu_runs = Path(args.gpu_runs_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    perf_cpu, res_cpu = _collect_rows("cpu", cpu_runs)
    perf_gpu, res_gpu = _collect_rows("gpu", gpu_runs)
    perf_df = pd.DataFrame(perf_cpu + perf_gpu)
    res_df = pd.DataFrame(res_cpu + res_gpu)

    perf_csv = out_dir / "formal_performance_table_2026-03-05.csv"
    res_csv = out_dir / "formal_resource_table_2026-03-05.csv"
    perf_df.to_csv(perf_csv, index=False)
    res_df.to_csv(res_csv, index=False)

    _write_markdown_table(out_dir / "formal_performance_table_2026-03-05.md", "Formal Performance Table", perf_df)
    _write_markdown_table(out_dir / "formal_resource_table_2026-03-05.md", "Formal Resource Table", res_df)

    print(str(perf_csv))
    print(str(res_csv))


if __name__ == "__main__":
    main()
