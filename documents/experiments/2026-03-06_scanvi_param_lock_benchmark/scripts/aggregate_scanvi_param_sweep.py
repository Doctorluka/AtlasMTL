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
        default="/tmp/atlasmtl_benchmarks/2026-03-06/scanvi_param_lock",
    )
    parser.add_argument(
        "--results-dir",
        default="documents/experiments/2026-03-06_scanvi_param_lock_benchmark/results_summary",
    )
    parser.add_argument("--top-k", type=int, default=2)
    return parser.parse_args()


def _load_stage_csv(stage_root: Path) -> pd.DataFrame:
    csv_path = stage_root / "run_index.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    return df


def _safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _stage_a_ranking(stage_a: pd.DataFrame) -> pd.DataFrame:
    if stage_a.empty:
        return stage_a
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
            "scvi_max_epochs",
            "scanvi_max_epochs",
            "query_max_epochs",
            "n_latent",
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
            scvi_max_epochs=("scvi_max_epochs", "first"),
            scanvi_max_epochs=("scanvi_max_epochs", "first"),
            query_max_epochs=("query_max_epochs", "first"),
            n_latent=("n_latent", "first"),
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
        return stage_b
    use = stage_b[stage_b["success"] == True].copy()  # noqa: E712
    if use.empty:
        return pd.DataFrame()
    use = _safe_numeric(use, ["accuracy", "macro_f1", "query_size", "seed"])
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


def _select_top_params(stage_a_rank: pd.DataFrame, top_k: int) -> List[Dict]:
    if stage_a_rank.empty:
        return []
    top = stage_a_rank.head(top_k).copy()
    items: List[Dict] = []
    for _, row in top.iterrows():
        items.append(
            {
                "param_id": row["param_id"],
                "scvi_max_epochs": int(row["scvi_max_epochs"]),
                "scanvi_max_epochs": int(row["scanvi_max_epochs"]),
                "query_max_epochs": int(row["query_max_epochs"]),
                "n_latent": int(row["n_latent"]),
                "mean_macro_f1": float(row["mean_macro_f1"]),
                "mean_accuracy": float(row["mean_accuracy"]),
                "mean_total_elapsed_seconds": float(row["mean_total_elapsed_seconds"]),
            }
        )
    return items


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    stage_a = _load_stage_csv(output_root / "stage_a")
    stage_b = _load_stage_csv(output_root / "stage_b")

    all_runs = pd.concat([x for x in [stage_a, stage_b] if not x.empty], ignore_index=True) if (not stage_a.empty or not stage_b.empty) else pd.DataFrame()
    stage_a_rank = _stage_a_ranking(stage_a)
    stage_b_stability = _stage_b_stability(stage_b)

    raw_csv = results_dir / "sweep_raw_results.csv"
    rank_csv = results_dir / "stage_a_param_ranking.csv"
    stbl_csv = results_dir / "stage_b_stability.csv"
    if not all_runs.empty:
        all_runs.to_csv(raw_csv, index=False)
    else:
        raw_csv.write_text("", encoding="utf-8")
    if not stage_a_rank.empty:
        stage_a_rank.to_csv(rank_csv, index=False)
    else:
        rank_csv.write_text("", encoding="utf-8")
    if not stage_b_stability.empty:
        stage_b_stability.to_csv(stbl_csv, index=False)
    else:
        stbl_csv.write_text("", encoding="utf-8")

    top_params = _select_top_params(stage_a_rank, int(args.top_k))
    top_payload = {
        "selection_rule": {
            "primary": "mean_macro_f1 desc",
            "secondary": "mean_accuracy desc",
            "tiebreak": "mean_total_elapsed_seconds asc",
        },
        "top_params": top_params,
    }
    top_json_results = results_dir / "top2_params.json"
    top_json_stage = output_root / "stage_a" / "top2_params.json"
    top_json_results.write_text(json.dumps(top_payload, indent=2, sort_keys=True), encoding="utf-8")
    top_json_stage.parent.mkdir(parents=True, exist_ok=True)
    top_json_stage.write_text(json.dumps(top_payload, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "raw_csv": str(raw_csv),
        "stage_a_param_ranking_csv": str(rank_csv),
        "stage_b_stability_csv": str(stbl_csv),
        "top_params_json_results": str(top_json_results),
        "top_params_json_stage_a": str(top_json_stage),
        "n_stage_a_runs": int(len(stage_a)) if not stage_a.empty else 0,
        "n_stage_b_runs": int(len(stage_b)) if not stage_b.empty else 0,
        "n_top_params": int(len(top_params)),
    }
    (results_dir / "aggregation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
