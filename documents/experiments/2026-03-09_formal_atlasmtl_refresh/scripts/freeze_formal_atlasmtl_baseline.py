#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
ROUND_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-09_formal_atlasmtl_refresh"
RESULTS_DIR = ROUND_ROOT / "results_summary"
FORMAL_SUMMARY = (
    REPO_ROOT
    / "documents"
    / "experiments"
    / "2026-03-06_formal_third_wave_scaling"
    / "results_summary"
)

PERF_PATH = FORMAL_SUMMARY / "formal_comparative_performance_snapshot_2026-03-09.csv"
RESOURCE_PATH = FORMAL_SUMMARY / "formal_comparative_resource_snapshot_2026-03-09.csv"
SCOPE_PATH = FORMAL_SUMMARY / "formal_result_scope_map_2026-03-09.csv"

KEEP_POINTS = {
    ("HLCA_Core", "cpu_core", "build_100000_eval10k"),
    ("HLCA_Core", "cpu_core", "predict_100000_10000"),
    ("HLCA_Core", "gpu", "build_100000_eval10k"),
    ("HLCA_Core", "gpu", "predict_100000_10000"),
    ("PHMap_Lung_Full_v43_light", "cpu_core", "build_100000_eval10k"),
    ("PHMap_Lung_Full_v43_light", "cpu_core", "predict_100000_10000"),
    ("PHMap_Lung_Full_v43_light", "gpu", "build_100000_eval10k"),
    ("PHMap_Lung_Full_v43_light", "gpu", "predict_100000_10000"),
    ("mTCA", "cpu_core", "build_100000_eval10k"),
    ("mTCA", "cpu_core", "predict_100000_10000"),
    ("mTCA", "gpu", "build_100000_eval10k"),
    ("mTCA", "gpu", "predict_100000_10000"),
    ("DISCO_hPBMCs", "cpu_core", "build_100000_eval10k"),
    ("DISCO_hPBMCs", "cpu_core", "predict_100000_10000"),
    ("DISCO_hPBMCs", "gpu", "build_100000_eval10k"),
    ("DISCO_hPBMCs", "gpu", "predict_100000_10000"),
    ("Vento", "cpu_core", "build_50000_eval10k"),
    ("Vento", "cpu_core", "predict_50000_10000"),
    ("Vento", "gpu", "build_50000_eval10k"),
    ("Vento", "gpu", "predict_50000_10000"),
}


def main() -> None:
    perf = pd.read_csv(PERF_PATH)
    resource = pd.read_csv(RESOURCE_PATH)
    scope = pd.read_csv(SCOPE_PATH)

    perf = perf[perf["method"] == "atlasmtl"].copy()
    resource = resource[resource["method"] == "atlasmtl"].copy()
    scope = scope[scope["method"] == "atlasmtl"].copy()

    key_cols = ["dataset", "track", "point", "method"]
    merged = perf.merge(resource, on=key_cols, how="left").merge(scope, on=key_cols, how="left")
    merged["keep"] = merged.apply(lambda r: (r["dataset"], r["track"], r["point"]) in KEEP_POINTS, axis=1)
    kept = merged[merged["keep"]].copy()
    if len(kept) != 20:
        raise ValueError(f"expected 20 formal atlasmtl anchor rows, found {len(kept)}")

    kept["baseline_snapshot_source"] = PERF_PATH.resolve().as_posix()
    kept["resource_snapshot_source"] = RESOURCE_PATH.resolve().as_posix()
    kept["scope_map_source"] = SCOPE_PATH.resolve().as_posix()

    rename_map = {
        "accuracy": "old_accuracy",
        "macro_f1": "old_macro_f1",
        "balanced_accuracy": "old_balanced_accuracy",
        "risk": "old_risk",
        "train_elapsed_seconds": "old_train_elapsed_seconds",
        "predict_elapsed_seconds": "old_predict_elapsed_seconds",
        "train_process_peak_rss_gb": "old_train_process_peak_rss_gb",
        "predict_process_peak_rss_gb": "old_predict_process_peak_rss_gb",
        "train_gpu_peak_memory_gb": "old_train_gpu_peak_memory_gb",
        "predict_gpu_peak_memory_gb": "old_predict_gpu_peak_memory_gb",
    }
    kept = kept.rename(columns=rename_map)
    keep_cols = [
        "dataset",
        "track",
        "point",
        "scope",
        "display_policy",
        "notes",
        "old_accuracy",
        "old_macro_f1",
        "old_balanced_accuracy",
        "old_risk",
        "old_train_elapsed_seconds",
        "old_predict_elapsed_seconds",
        "old_train_process_peak_rss_gb",
        "old_predict_process_peak_rss_gb",
        "old_train_gpu_peak_memory_gb",
        "old_predict_gpu_peak_memory_gb",
        "baseline_snapshot_source",
        "resource_snapshot_source",
        "scope_map_source",
    ]
    kept = kept[keep_cols].sort_values(["dataset", "track", "point"]).reset_index(drop=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "atlasmtl_formal_baseline_anchor.csv"
    kept.to_csv(out_csv, index=False)
    print(json.dumps({"row_count": len(kept), "output": out_csv.resolve().as_posix()}, indent=2))


if __name__ == "__main__":
    main()
