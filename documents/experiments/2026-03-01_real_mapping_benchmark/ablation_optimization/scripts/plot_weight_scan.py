#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ablation_common import aggregate_runtime_peak, first_hierarchy_edge_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate task-weight scan tables and line plots.")
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--output-dir", help="Defaults to <metrics_json_dir>/analysis")
    return parser.parse_args()


def _load_metrics(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "results" not in payload:
        raise ValueError("metrics.json must contain top-level 'results'")
    return payload


def _rows(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for result in results:
        ablation = result.get("ablation_config") or {}
        metrics = result.get("metrics") or {}
        lv4 = metrics.get("anno_lv4") or {}
        hierarchy = result.get("hierarchy_metrics") or {}
        quality_score = 0.5 * float(lv4.get("accuracy") or 0.0) + 0.5 * float(lv4.get("macro_f1") or 0.0)
        rows.append(
            {
                "variant_name": result.get("variant_name"),
                "device": ablation.get("device"),
                "feature_mode": ablation.get("feature_mode"),
                "task_weight_scheme": ablation.get("task_weight_scheme"),
                "weight_param_r": ablation.get("weight_param_r"),
                "task_weights": json.dumps(ablation.get("task_weights") or []),
                "is_ratio_scan": ablation.get("weight_param_r") is not None,
                "lv1_accuracy": (metrics.get("anno_lv1") or {}).get("accuracy"),
                "lv2_accuracy": (metrics.get("anno_lv2") or {}).get("accuracy"),
                "lv3_accuracy": (metrics.get("anno_lv3") or {}).get("accuracy"),
                "lv4_accuracy": lv4.get("accuracy"),
                "lv4_macro_f1": lv4.get("macro_f1"),
                "full_path_accuracy": hierarchy.get("full_path_accuracy"),
                "full_path_coverage": hierarchy.get("full_path_coverage"),
                "hierarchy_consistency": first_hierarchy_edge_rate(result),
                "train_elapsed_seconds": (result.get("train_usage") or {}).get("elapsed_seconds"),
                "predict_elapsed_seconds": (result.get("predict_usage") or {}).get("elapsed_seconds"),
                "peak_rss_gb": aggregate_runtime_peak(result, "process_peak_rss_gb"),
                "peak_gpu_memory_gb": aggregate_runtime_peak(result, "gpu_peak_memory_gb"),
                "quality_score": quality_score,
            }
        )
    return pd.DataFrame(rows).sort_values(["device", "weight_param_r", "task_weight_scheme"]).reset_index(drop=True)


def _baseline_row(group: pd.DataFrame, scheme_name: str) -> pd.Series | None:
    subset = group[group["task_weight_scheme"] == scheme_name]
    if subset.empty:
        return None
    return subset.iloc[0]


def _recommendations(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for device, group in df.groupby("device", sort=False):
        best_quality = float(group["quality_score"].max())
        near_opt = group[group["quality_score"] >= 0.95 * best_quality].copy()
        uniform = _baseline_row(group, "uniform")
        if uniform is not None:
            near_opt["coarse_regression_flag"] = (
                (near_opt["lv1_accuracy"] < float(uniform["lv1_accuracy"]) - 0.01)
                | (near_opt["lv2_accuracy"] < float(uniform["lv2_accuracy"]) - 0.01)
                | (near_opt["full_path_accuracy"] < float(uniform["full_path_accuracy"]) - 0.01)
                | (near_opt["full_path_coverage"] < float(uniform["full_path_coverage"]) - 0.01)
            )
            eligible = near_opt[~near_opt["coarse_regression_flag"]].copy()
            if eligible.empty:
                eligible = near_opt.copy()
        else:
            near_opt["coarse_regression_flag"] = False
            eligible = near_opt.copy()
        eligible = eligible.sort_values(
            ["peak_rss_gb", "train_elapsed_seconds", "predict_elapsed_seconds", "peak_gpu_memory_gb"],
            na_position="last",
        )
        recommended = eligible.iloc[0].to_dict()
        recommended["selection_rule"] = "quality_score >= 95% of best, no major coarse/full-path regression when possible"
        recommended["best_quality_score"] = best_quality
        rows.append(recommended)
    return pd.DataFrame(rows)


def _plot_metric(df: pd.DataFrame, *, metric: str, title: str, ylabel: str, output_path: Path) -> None:
    ratio_df = df[df["is_ratio_scan"]].copy()
    if ratio_df.empty:
        return
    fig, axes = plt.subplots(1, len(ratio_df["device"].dropna().unique()), figsize=(12, 4), sharey=False)
    if hasattr(axes, "ravel"):
        axes = list(axes.ravel())
    else:
        axes = [axes]
    for axis, (device, group) in zip(axes, ratio_df.groupby("device", sort=False)):
        group = group.sort_values("weight_param_r")
        axis.plot(group["weight_param_r"], group[metric], marker="o", linewidth=2, label="ratio scan")
        anchors = df[(df["device"] == device) & (~df["is_ratio_scan"])]
        for _, row in anchors.iterrows():
            if pd.notna(row[metric]):
                axis.axhline(float(row[metric]), linestyle="--", linewidth=1.2, label=str(row["task_weight_scheme"]))
        axis.set_title(str(device))
        axis.set_xlabel("weight ratio r")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        dedup: Dict[str, Any] = {}
        for handle, label in zip(handles, labels):
            dedup[label] = handle
        axis.legend(dedup.values(), dedup.keys(), fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_markdown(recommendations: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Weight Scan Recommendation",
        "",
        "This is an internal benchmark note for the current optimization task.",
        "It is not the final paper-facing conclusion.",
        "",
    ]
    for _, row in recommendations.iterrows():
        lines.extend(
            [
                f"## {row['device']}",
                "",
                f"- recommended scheme: `{row['task_weight_scheme']}`",
                f"- task weights: `{row['task_weights']}`",
                f"- `weight_param_r = {row['weight_param_r']}`" if pd.notna(row["weight_param_r"]) else "- `weight_param_r = anchor`",
                f"- `lv4_accuracy = {row['lv4_accuracy']:.4f}`",
                f"- `lv4_macro_f1 = {row['lv4_macro_f1']:.4f}`",
                f"- `full_path_accuracy = {row['full_path_accuracy']:.4f}`",
                f"- `peak_rss_gb = {row['peak_rss_gb']:.4f}`",
                f"- `train_elapsed_seconds = {row['train_elapsed_seconds']:.4f}`",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics_json).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else metrics_path.parent / "analysis"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_metrics(metrics_path)
    dataframe = _rows(payload["results"])
    dataframe.to_csv(output_dir / "atlasmtl_weight_scan.csv", index=False)
    recommendations = _recommendations(dataframe)
    recommendations.to_csv(output_dir / "atlasmtl_weight_recommendations.csv", index=False)
    _write_markdown(recommendations, output_dir / "weight_scan_recommendation.md")

    plot_specs = [
        ("lv4_accuracy", "LV4 accuracy vs weight ratio", "lv4 accuracy", "lv4_accuracy_vs_weight_r.png"),
        ("lv4_macro_f1", "LV4 macro-F1 vs weight ratio", "lv4 macro-F1", "lv4_macro_f1_vs_weight_r.png"),
        ("full_path_accuracy", "Full-path accuracy vs weight ratio", "full-path accuracy", "full_path_accuracy_vs_weight_r.png"),
        ("lv1_accuracy", "LV1 accuracy vs weight ratio", "lv1 accuracy", "lv1_accuracy_vs_weight_r.png"),
        ("lv2_accuracy", "LV2 accuracy vs weight ratio", "lv2 accuracy", "lv2_accuracy_vs_weight_r.png"),
        ("train_elapsed_seconds", "Train time vs weight ratio", "train seconds", "train_time_vs_weight_r.png"),
        ("predict_elapsed_seconds", "Predict time vs weight ratio", "predict seconds", "predict_time_vs_weight_r.png"),
        ("peak_rss_gb", "Peak RSS vs weight ratio", "peak RSS (GB)", "peak_rss_vs_weight_r.png"),
        ("peak_gpu_memory_gb", "Peak GPU memory vs weight ratio", "peak GPU memory (GB)", "peak_gpu_mem_vs_weight_r.png"),
    ]
    for metric, title, ylabel, filename in plot_specs:
        if dataframe[metric].notna().any():
            _plot_metric(dataframe, metric=metric, title=title, ylabel=ylabel, output_path=figures_dir / filename)

    print(output_dir)


if __name__ == "__main__":
    main()
