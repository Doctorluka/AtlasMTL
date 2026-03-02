#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


MAIN_METRICS = [
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "coverage",
    "reject_rate",
    "ece",
    "brier",
    "aurc",
]


def _effective_variant_name(result: Dict[str, Any]) -> Any:
    ablation = result.get("ablation_config") or {}
    return ablation.get("variant_name") or result.get("variant_name")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export paper-ready benchmark tables from metrics.json.",
    )
    parser.add_argument("--metrics-json", required=True, help="Path to benchmark metrics.json")
    parser.add_argument(
        "--output-dir",
        help="Directory for exported tables. Defaults to <metrics_json_dir>/paper_tables",
    )
    parser.add_argument(
        "--target-label-column",
        help="Prefer a specific label column when methods expose multiple levels",
    )
    return parser.parse_args()


def _load_metrics(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "results" not in payload:
        raise ValueError("metrics.json must contain a top-level 'results' field")
    if not isinstance(payload["results"], list):
        raise ValueError("'results' must be a list")
    return payload


def _first_level(metrics: Dict[str, Any], preferred: str | None) -> Tuple[str | None, Dict[str, Any]]:
    if not isinstance(metrics, dict) or not metrics:
        return None, {}
    if preferred and preferred in metrics:
        value = metrics.get(preferred)
        return preferred, value if isinstance(value, dict) else {}
    level, value = next(iter(metrics.items()))
    return str(level), value if isinstance(value, dict) else {}


def _main_table(results: List[Dict[str, Any]], target_label_column: str | None) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for result in results:
        level, level_metrics = _first_level(result.get("metrics", {}), target_label_column)
        _, behavior_metrics = _first_level(result.get("behavior_metrics", {}), target_label_column)
        metadata = result.get("prediction_metadata") or {}
        row: Dict[str, Any] = {
            "method": result.get("method"),
            "variant_name": _effective_variant_name(result),
            "level": level,
            "backend": metadata.get("implementation_backend") or metadata.get("comparator_name") or result.get("method"),
            "knn_space_used": metadata.get("knn_space_used"),
            "knn_space_preferred": metadata.get("knn_space_preferred"),
        }
        for metric in MAIN_METRICS:
            row[metric] = level_metrics.get(metric)
        row["unknown_rate"] = behavior_metrics.get("unknown_rate")
        rows.append(row)
    return pd.DataFrame(rows)


def _domain_table(results: List[Dict[str, Any]], target_label_column: str | None) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for result in results:
        metrics_by_domain = result.get("metrics_by_domain") or {}
        behavior_by_domain = result.get("behavior_metrics_by_domain") or {}
        for domain, domain_metrics in metrics_by_domain.items():
            level, level_metrics = _first_level(domain_metrics, target_label_column)
            _, behavior_metrics = _first_level(behavior_by_domain.get(domain, {}), target_label_column)
            rows.append(
                {
                    "method": result.get("method"),
                    "variant_name": _effective_variant_name(result),
                    "domain": domain,
                    "level": level,
                    "accuracy": level_metrics.get("accuracy"),
                    "macro_f1": level_metrics.get("macro_f1"),
                    "balanced_accuracy": level_metrics.get("balanced_accuracy"),
                    "coverage": level_metrics.get("coverage"),
                    "reject_rate": level_metrics.get("reject_rate"),
                    "unknown_rate": behavior_metrics.get("unknown_rate"),
                }
            )
    return pd.DataFrame(rows)


def _atlasmtl_table(results: List[Dict[str, Any]], target_label_column: str | None) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for atlas in results:
        if atlas.get("method") != "atlasmtl":
            continue
        level, behavior_metrics = _first_level(atlas.get("behavior_metrics", {}), target_label_column)
        hierarchy_metrics = atlas.get("hierarchy_metrics") or {}
        rows.extend(
            [
                {
                    "variant_name": _effective_variant_name(atlas),
                    "metric": "unknown_rate",
                    "value": behavior_metrics.get("unknown_rate"),
                    "level": level,
                },
                {
                    "variant_name": _effective_variant_name(atlas),
                    "metric": "knn_coverage",
                    "value": behavior_metrics.get("knn_coverage"),
                    "level": level,
                },
                {
                    "variant_name": _effective_variant_name(atlas),
                    "metric": "knn_rescue_rate",
                    "value": behavior_metrics.get("knn_rescue_rate"),
                    "level": level,
                },
                {
                    "variant_name": _effective_variant_name(atlas),
                    "metric": "knn_harm_rate",
                    "value": behavior_metrics.get("knn_harm_rate"),
                    "level": level,
                },
                {
                    "variant_name": _effective_variant_name(atlas),
                    "metric": "full_path_accuracy",
                    "value": hierarchy_metrics.get("full_path_accuracy"),
                    "level": level,
                },
                {
                    "variant_name": _effective_variant_name(atlas),
                    "metric": "path_consistency_rate",
                    "value": hierarchy_metrics.get("path_consistency_rate"),
                    "level": level,
                },
            ]
        )
    return pd.DataFrame(rows)


def _coordinate_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for result in results:
        coord_metrics = result.get("coordinate_metrics") or {}
        if not coord_metrics:
            continue
        row = {"method": result.get("method")}
        row.update(coord_metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def _protocol_table(results: List[Dict[str, Any]], target_label_column: str | None) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for result in results:
        label_columns = list(result.get("label_columns") or [])
        resolved_label = target_label_column or (label_columns[-1] if label_columns else None)
        contract = result.get("input_contract") or {}
        rows.append(
            {
                "method": result.get("method"),
                "variant_name": _effective_variant_name(result),
                "backend": contract.get("backend"),
                "target_label_column": resolved_label,
                "label_scope": contract.get("label_scope"),
                "reference_matrix_source": contract.get("reference_matrix_source"),
                "query_matrix_source": contract.get("query_matrix_source"),
                "counts_layer": contract.get("counts_layer"),
                "normalization_mode": contract.get("normalization_mode"),
                "feature_alignment": contract.get("feature_alignment"),
                "knn_space_used": (result.get("prediction_metadata") or {}).get("knn_space_used"),
                "knn_space_preferred": (result.get("prediction_metadata") or {}).get("knn_space_preferred"),
            }
        )
    return pd.DataFrame(rows)


def _runtime_resource_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for result in results:
        train_usage = result.get("train_usage") or {}
        predict_usage = result.get("predict_usage") or {}
        rows.append(
            {
                "method": result.get("method"),
                "variant_name": _effective_variant_name(result),
                "backend": (result.get("input_contract") or {}).get("backend"),
                "device_used": predict_usage.get("device_used") or train_usage.get("device_used"),
                "num_threads_used": predict_usage.get("num_threads_used") or train_usage.get("num_threads_used"),
                "train_elapsed_seconds": train_usage.get("elapsed_seconds"),
                "predict_elapsed_seconds": predict_usage.get("elapsed_seconds"),
                "train_items_per_second": train_usage.get("items_per_second"),
                "predict_items_per_second": predict_usage.get("items_per_second"),
                "train_cpu_percent_avg": train_usage.get("cpu_percent_avg"),
                "predict_cpu_percent_avg": predict_usage.get("cpu_percent_avg"),
                "train_cpu_core_equiv_avg": train_usage.get("cpu_core_equiv_avg"),
                "predict_cpu_core_equiv_avg": predict_usage.get("cpu_core_equiv_avg"),
                "train_process_avg_rss_gb": train_usage.get("process_avg_rss_gb"),
                "predict_process_avg_rss_gb": predict_usage.get("process_avg_rss_gb"),
                "train_process_peak_rss_gb": train_usage.get("process_peak_rss_gb"),
                "predict_process_peak_rss_gb": predict_usage.get("process_peak_rss_gb"),
                "train_gpu_avg_memory_gb": train_usage.get("gpu_avg_memory_gb"),
                "predict_gpu_avg_memory_gb": predict_usage.get("gpu_avg_memory_gb"),
                "train_gpu_peak_memory_gb": train_usage.get("gpu_peak_memory_gb"),
                "predict_gpu_peak_memory_gb": predict_usage.get("gpu_peak_memory_gb"),
            }
        )
    return pd.DataFrame(rows)


def _atlasmtl_ablation_accuracy_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for result in results:
        ablation = result.get("ablation_config") or {}
        if result.get("method") != "atlasmtl" or not ablation:
            continue
        row: Dict[str, Any] = {
            "method": result.get("method"),
            "variant_name": _effective_variant_name(result),
            "device": ablation.get("device"),
            "feature_space": ablation.get("feature_space"),
            "n_top_genes": ablation.get("n_top_genes"),
            "input_transform": ablation.get("input_transform"),
            "task_weight_scheme": ablation.get("task_weight_scheme"),
            "knn_variant": ablation.get("knn_variant"),
            "knn_correction": ablation.get("knn_correction"),
            "knn_space_used": (result.get("prediction_metadata") or {}).get("knn_space_used"),
        }
        for level, metrics in (result.get("metrics") or {}).items():
            row[f"{level}_accuracy"] = (metrics or {}).get("accuracy")
            row[f"{level}_macro_f1"] = (metrics or {}).get("macro_f1")
            row[f"{level}_balanced_accuracy"] = (metrics or {}).get("balanced_accuracy")
        hierarchy_metrics = result.get("hierarchy_metrics") or {}
        first_hierarchy = next(iter(hierarchy_metrics.values()), {}) if hierarchy_metrics else {}
        row["path_consistency_rate"] = (first_hierarchy or {}).get("path_consistency_rate")
        row["full_path_accuracy"] = (first_hierarchy or {}).get("full_path_accuracy")
        rows.append(row)
    return pd.DataFrame(rows)


def _atlasmtl_ablation_resource_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for result in results:
        ablation = result.get("ablation_config") or {}
        if result.get("method") != "atlasmtl" or not ablation:
            continue
        train_usage = result.get("train_usage") or {}
        predict_usage = result.get("predict_usage") or {}
        rows.append(
            {
                "method": result.get("method"),
                "variant_name": _effective_variant_name(result),
                "device": ablation.get("device"),
                "feature_space": ablation.get("feature_space"),
                "n_top_genes": ablation.get("n_top_genes"),
                "input_transform": ablation.get("input_transform"),
                "task_weight_scheme": ablation.get("task_weight_scheme"),
                "knn_variant": ablation.get("knn_variant"),
                "knn_correction": ablation.get("knn_correction"),
                "train_elapsed_seconds": train_usage.get("elapsed_seconds"),
                "predict_elapsed_seconds": predict_usage.get("elapsed_seconds"),
                "train_process_avg_rss_gb": train_usage.get("process_avg_rss_gb"),
                "predict_process_avg_rss_gb": predict_usage.get("process_avg_rss_gb"),
                "train_process_peak_rss_gb": train_usage.get("process_peak_rss_gb"),
                "predict_process_peak_rss_gb": predict_usage.get("process_peak_rss_gb"),
                "train_gpu_avg_memory_gb": train_usage.get("gpu_avg_memory_gb"),
                "predict_gpu_avg_memory_gb": predict_usage.get("gpu_avg_memory_gb"),
                "train_gpu_peak_memory_gb": train_usage.get("gpu_peak_memory_gb"),
                "predict_gpu_peak_memory_gb": predict_usage.get("gpu_peak_memory_gb"),
                "train_cpu_core_equiv_avg": train_usage.get("cpu_core_equiv_avg"),
                "predict_cpu_core_equiv_avg": predict_usage.get("cpu_core_equiv_avg"),
            }
        )
    return pd.DataFrame(rows)


def _atlasmtl_ablation_tradeoff_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for result in results:
        ablation = result.get("ablation_config") or {}
        metrics = result.get("metrics") or {}
        if result.get("method") != "atlasmtl" or not ablation:
            continue
        lv4 = metrics.get("anno_lv4")
        if lv4 is None and metrics:
            lv4 = next(reversed(list(metrics.values())))
        train_usage = result.get("train_usage") or {}
        predict_usage = result.get("predict_usage") or {}
        rows.append(
            {
                "variant_name": _effective_variant_name(result),
                "device": ablation.get("device"),
                "feature_space": ablation.get("feature_space"),
                "n_top_genes": ablation.get("n_top_genes"),
                "input_transform": ablation.get("input_transform"),
                "task_weight_scheme": ablation.get("task_weight_scheme"),
                "knn_variant": ablation.get("knn_variant"),
                "knn_correction": ablation.get("knn_correction"),
                "knn_space_used": (result.get("prediction_metadata") or {}).get("knn_space_used"),
                "target_level_accuracy": (lv4 or {}).get("accuracy"),
                "target_level_macro_f1": (lv4 or {}).get("macro_f1"),
                "train_elapsed_seconds": train_usage.get("elapsed_seconds"),
                "predict_elapsed_seconds": predict_usage.get("elapsed_seconds"),
                "train_process_peak_rss_gb": train_usage.get("process_peak_rss_gb"),
                "predict_process_peak_rss_gb": predict_usage.get("process_peak_rss_gb"),
                "train_gpu_peak_memory_gb": train_usage.get("gpu_peak_memory_gb"),
                "predict_gpu_peak_memory_gb": predict_usage.get("gpu_peak_memory_gb"),
            }
        )
    return pd.DataFrame(rows)


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format_value(row[col]) for col in df.columns) + " |")
    return "\n".join(lines)


def _write_markdown(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        path.write_text("", encoding="utf-8")
        return
    path.write_text(_dataframe_to_markdown(df), encoding="utf-8")


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics_json).resolve()
    payload = _load_metrics(metrics_path)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else metrics_path.parent / "paper_tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    main_df = _main_table(payload["results"], args.target_label_column)
    domain_df = _domain_table(payload["results"], args.target_label_column)
    atlas_df = _atlasmtl_table(payload["results"], args.target_label_column)
    coordinate_df = _coordinate_table(payload["results"])
    protocol_df = _protocol_table(payload["results"], args.target_label_column)
    runtime_df = _runtime_resource_table(payload["results"])
    atlas_ablation_accuracy_df = _atlasmtl_ablation_accuracy_table(payload["results"])
    atlas_ablation_resources_df = _atlasmtl_ablation_resource_table(payload["results"])
    atlas_ablation_tradeoff_df = _atlasmtl_ablation_tradeoff_table(payload["results"])

    tables = {
        "main_comparison": main_df,
        "comparator_protocol": protocol_df,
        "runtime_resources": runtime_df,
        "domain_comparison": domain_df,
        "atlasmtl_analysis": atlas_df,
        "coordinate_diagnostics": coordinate_df,
        "atlasmtl_ablation_accuracy": atlas_ablation_accuracy_df,
        "atlasmtl_ablation_resources": atlas_ablation_resources_df,
        "atlasmtl_ablation_tradeoff": atlas_ablation_tradeoff_df,
    }

    for name, dataframe in tables.items():
        dataframe.to_csv(output_dir / f"{name}.csv", index=False)
        _write_markdown(dataframe, output_dir / f"{name}.md")

    manifest = {
        "metrics_json": str(metrics_path),
        "output_dir": str(output_dir),
        "target_label_column": args.target_label_column,
        "tables": {name: {"rows": int(len(df)), "empty": bool(df.empty)} for name, df in tables.items()},
    }
    (output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
