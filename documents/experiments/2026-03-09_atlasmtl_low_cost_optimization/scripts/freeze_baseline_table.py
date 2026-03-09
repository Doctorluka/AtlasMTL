#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
ROUND_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-09_atlasmtl_low_cost_optimization"
RESULTS_DIR = ROUND_ROOT / "results_summary"
PERF_CSV = (
    REPO_ROOT
    / "documents"
    / "experiments"
    / "2026-03-06_formal_third_wave_scaling"
    / "results_summary"
    / "formal_comparative_performance_snapshot_2026-03-09.csv"
)
RESOURCE_CSV = (
    REPO_ROOT
    / "documents"
    / "experiments"
    / "2026-03-06_formal_third_wave_scaling"
    / "results_summary"
    / "formal_comparative_resource_snapshot_2026-03-09.csv"
)

SELECTED_DATASETS = {"PHMap_Lung_Full_v43_light", "mTCA"}
SELECTED_POINTS = {"build_100000_eval10k", "predict_100000_10000"}
SELECTED_TRACKS = {"cpu_core", "gpu"}


def main() -> None:
    perf = pd.read_csv(PERF_CSV)
    resource = pd.read_csv(RESOURCE_CSV)

    perf = perf[
        perf["dataset"].isin(SELECTED_DATASETS)
        & perf["point"].isin(SELECTED_POINTS)
        & perf["track"].isin(SELECTED_TRACKS)
        & (perf["method"] == "atlasmtl")
    ].copy()
    resource = resource[
        resource["dataset"].isin(SELECTED_DATASETS)
        & resource["point"].isin(SELECTED_POINTS)
        & resource["track"].isin(SELECTED_TRACKS)
        & (resource["method"] == "atlasmtl")
    ].copy()

    merged = perf.merge(
        resource[
            [
                "dataset",
                "track",
                "point",
                "method",
                "train_elapsed_seconds",
                "predict_elapsed_seconds",
                "train_process_peak_rss_gb",
                "predict_process_peak_rss_gb",
                "train_gpu_peak_memory_gb",
                "predict_gpu_peak_memory_gb",
            ]
        ],
        on=["dataset", "track", "point", "method"],
        how="left",
    )
    merged.insert(3, "config_name", "baseline_anchor")
    merged = merged[
        [
            "dataset",
            "track",
            "point",
            "config_name",
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "train_elapsed_seconds",
            "predict_elapsed_seconds",
            "train_process_peak_rss_gb",
            "predict_process_peak_rss_gb",
            "train_gpu_peak_memory_gb",
            "predict_gpu_peak_memory_gb",
        ]
    ].sort_values(["dataset", "track", "point"]).reset_index(drop=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "stage_a_baseline_anchor.csv"
    merged.to_csv(out_csv, index=False)
    print(json.dumps({"rows": int(len(merged)), "output_csv": str(out_csv.resolve())}, indent=2))


if __name__ == "__main__":
    main()
