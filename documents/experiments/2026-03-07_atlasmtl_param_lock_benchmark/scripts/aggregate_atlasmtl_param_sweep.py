#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="/tmp/atlasmtl_benchmarks/2026-03-07/atlasmtl_param_lock",
    )
    parser.add_argument(
        "--results-dir",
        default="documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/results_summary",
    )
    parser.add_argument("--top-k", type=int, default=2)
    return parser.parse_args()


def _load_run_index(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


def _safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _stage_a_ranking(stage_a: pd.DataFrame) -> pd.DataFrame:
    if stage_a.empty:
        return pd.DataFrame()
    use = stage_a[stage_a["success"] == True].copy()  # noqa: E712
    if use.empty:
        return pd.DataFrame()
    use = _safe_numeric(
        use,
        [
            "accuracy",
            "macro_f1",
            "train_elapsed_seconds",
            "predict_elapsed_seconds",
            "learning_rate",
            "batch_size",
        ],
    )
    use["total_elapsed_seconds"] = use["train_elapsed_seconds"].fillna(0.0) + use["predict_elapsed_seconds"].fillna(0.0)
    grouped = (
        use.groupby("param_id", dropna=False)
        .agg(
            n_runs=("param_id", "count"),
            n_datasets=("dataset_name", pd.Series.nunique),
            mean_macro_f1=("macro_f1", "mean"),
            mean_accuracy=("accuracy", "mean"),
            mean_total_elapsed_seconds=("total_elapsed_seconds", "mean"),
            learning_rate=("learning_rate", "first"),
            hidden_sizes=("hidden_sizes", "first"),
            batch_size=("batch_size", "first"),
        )
        .reset_index()
    )
    grouped = grouped.sort_values(
        by=["mean_macro_f1", "mean_accuracy", "mean_total_elapsed_seconds"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return grouped


def _stage_b_stability(stage_b: pd.DataFrame) -> pd.DataFrame:
    if stage_b.empty:
        return pd.DataFrame()
    use = stage_b[stage_b["success"] == True].copy()  # noqa: E712
    if use.empty:
        return pd.DataFrame()
    use = _safe_numeric(use, ["accuracy", "macro_f1", "query_size"])
    grouped = (
        use.groupby(["param_id", "query_size"], dropna=False)
        .agg(
            n_runs=("param_id", "count"),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
        )
        .reset_index()
        .sort_values(by=["query_size", "mean_macro_f1"], ascending=[True, False])
    )
    return grouped


def _select_top_params(rank_df: pd.DataFrame, top_k: int) -> List[Dict]:
    if rank_df.empty:
        return []
    top = rank_df.head(top_k)
    rows: List[Dict] = []
    for _, row in top.iterrows():
        rows.append(
            {
                "param_id": row["param_id"],
                "learning_rate": float(row["learning_rate"]),
                "hidden_sizes": str(row["hidden_sizes"]),
                "batch_size": int(row["batch_size"]),
                "mean_macro_f1": float(row["mean_macro_f1"]),
                "mean_accuracy": float(row["mean_accuracy"]),
                "mean_total_elapsed_seconds": float(row["mean_total_elapsed_seconds"]),
            }
        )
    return rows


def _write_top_params(payload: Dict, *, output_root: Path, results_dir: Path, device: str) -> None:
    results_path = results_dir / f"top2_params_{device}.json"
    stage_path = output_root / "stage_a" / device / "top2_params.json"
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    stage_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    stage_a_cpu = _load_run_index(output_root / "stage_a" / "cpu" / "run_index.csv")
    stage_a_gpu = _load_run_index(output_root / "stage_a" / "cuda" / "run_index.csv")
    stage_b_cpu = _load_run_index(output_root / "stage_b" / "cpu" / "run_index.csv")
    stage_b_gpu = _load_run_index(output_root / "stage_b" / "cuda" / "run_index.csv")

    def write_csv(df: pd.DataFrame, path: Path) -> None:
        if df.empty:
            path.write_text("", encoding="utf-8")
        else:
            df.to_csv(path, index=False)

    rank_cpu = _stage_a_ranking(stage_a_cpu)
    rank_gpu = _stage_a_ranking(stage_a_gpu)
    stbl_cpu = _stage_b_stability(stage_b_cpu)
    stbl_gpu = _stage_b_stability(stage_b_gpu)

    write_csv(rank_cpu, results_dir / "stage_a_core_ranking_cpu.csv")
    write_csv(rank_gpu, results_dir / "stage_a_core_ranking_gpu.csv")
    write_csv(stbl_cpu, results_dir / "stage_b_stability_cpu.csv")
    write_csv(stbl_gpu, results_dir / "stage_b_stability_gpu.csv")

    all_frames = [df for df in [stage_a_cpu, stage_a_gpu, stage_b_cpu, stage_b_gpu] if not df.empty]
    if all_frames:
        pd.concat(all_frames, ignore_index=True).to_csv(results_dir / "sweep_raw_results.csv", index=False)
    else:
        (results_dir / "sweep_raw_results.csv").write_text("", encoding="utf-8")

    payload_cpu = {
        "device": "cpu",
        "selection_rule": {
            "primary": "mean_macro_f1 desc",
            "secondary": "mean_accuracy desc",
            "tiebreak": "mean_total_elapsed_seconds asc",
        },
        "top_params": _select_top_params(rank_cpu, int(args.top_k)),
    }
    payload_gpu = {
        "device": "cuda",
        "selection_rule": {
            "primary": "mean_macro_f1 desc",
            "secondary": "mean_accuracy desc",
            "tiebreak": "mean_total_elapsed_seconds asc",
        },
        "top_params": _select_top_params(rank_gpu, int(args.top_k)),
    }
    _write_top_params(payload_cpu, output_root=output_root, results_dir=results_dir, device="cpu")
    _write_top_params(payload_gpu, output_root=output_root, results_dir=results_dir, device="cuda")

    lock_payload = {
        "cpu_default": (payload_cpu["top_params"][0] if payload_cpu["top_params"] else None),
        "gpu_default": (payload_gpu["top_params"][0] if payload_gpu["top_params"] else None),
        "cpu_candidates": payload_cpu["top_params"],
        "gpu_candidates": payload_gpu["top_params"],
    }
    (results_dir / "atlasmtl_locked_defaults.json").write_text(
        json.dumps(lock_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary = {
        "n_stage_a_cpu": int(len(stage_a_cpu)) if not stage_a_cpu.empty else 0,
        "n_stage_a_gpu": int(len(stage_a_gpu)) if not stage_a_gpu.empty else 0,
        "n_stage_b_cpu": int(len(stage_b_cpu)) if not stage_b_cpu.empty else 0,
        "n_stage_b_gpu": int(len(stage_b_gpu)) if not stage_b_gpu.empty else 0,
        "stage_a_core_ranking_cpu_csv": str(results_dir / "stage_a_core_ranking_cpu.csv"),
        "stage_a_core_ranking_gpu_csv": str(results_dir / "stage_a_core_ranking_gpu.csv"),
        "stage_b_stability_cpu_csv": str(results_dir / "stage_b_stability_cpu.csv"),
        "stage_b_stability_gpu_csv": str(results_dir / "stage_b_stability_gpu.csv"),
        "locked_defaults_json": str(results_dir / "atlasmtl_locked_defaults.json"),
    }
    (results_dir / "aggregation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
