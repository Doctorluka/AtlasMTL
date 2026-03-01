#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


MAIN_METRICS = (
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "coverage",
    "reject_rate",
    "ece",
    "brier",
    "aurc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a markdown benchmark summary from metrics.json.",
    )
    parser.add_argument("--metrics-json", required=True, help="Path to benchmark metrics.json")
    parser.add_argument(
        "--output",
        help="Output markdown path. Defaults to <metrics_json_dir>/benchmark_report.md",
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


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value:.4f}" if isinstance(value, float) else str(value)
    return str(value)


def _markdown_table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    headers_list = list(headers)
    header_row = "| " + " | ".join(headers_list) + " |"
    formatted_rows = []
    sep_row = "| " + " | ".join(["---"] * len(headers_list)) + " |"
    for row in rows:
        formatted_rows.append("| " + " | ".join(_format_value(item) for item in row) + " |")
    return "\n".join([header_row, sep_row, *formatted_rows])


def _collect_main_rows(results: List[Dict[str, Any]], target_label_column: str | None) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for result in results:
        level, level_metrics = _first_level(result.get("metrics", {}), target_label_column)
        behavior_level, behavior_metrics = _first_level(result.get("behavior_metrics", {}), target_label_column)
        prediction_metadata = result.get("prediction_metadata") or {}
        backend = prediction_metadata.get("implementation_backend") or prediction_metadata.get("comparator_name") or result.get("method")
        row = [
            result.get("method"),
            level or "",
        ]
        for metric_name in MAIN_METRICS:
            row.append(level_metrics.get(metric_name))
        row.extend(
            [
                behavior_metrics.get("unknown_rate") if level == behavior_level or behavior_level is None else None,
                backend,
            ]
        )
        rows.append(row)
    return rows


def _collect_domain_rows(results: List[Dict[str, Any]], target_label_column: str | None) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for result in results:
        metrics_by_domain = result.get("metrics_by_domain") or {}
        behavior_by_domain = result.get("behavior_metrics_by_domain") or {}
        for domain_name, domain_metrics in metrics_by_domain.items():
            level, level_metrics = _first_level(domain_metrics, target_label_column)
            _, behavior_metrics = _first_level(behavior_by_domain.get(domain_name, {}), target_label_column)
            rows.append(
                [
                    result.get("method"),
                    domain_name,
                    level or "",
                    level_metrics.get("accuracy"),
                    level_metrics.get("macro_f1"),
                    level_metrics.get("coverage"),
                    level_metrics.get("reject_rate"),
                    behavior_metrics.get("unknown_rate"),
                ]
            )
    return rows


def _collect_atlasmtl_rows(results: List[Dict[str, Any]], target_label_column: str | None) -> List[List[Any]]:
    atlas_result = next((item for item in results if item.get("method") == "atlasmtl"), None)
    if atlas_result is None:
        return []
    level, behavior = _first_level(atlas_result.get("behavior_metrics", {}), target_label_column)
    _, hierarchy = _first_level(atlas_result.get("hierarchy_metrics", {}), target_label_column)
    rows = [
        ["unknown_rate", behavior.get("unknown_rate"), ""],
        ["knn_coverage", behavior.get("knn_coverage"), ""],
        ["knn_rescue_rate", behavior.get("knn_rescue_rate"), ""],
        ["knn_harm_rate", behavior.get("knn_harm_rate"), ""],
        ["full_path_accuracy", hierarchy.get("full_path_accuracy"), level or ""],
        ["path_consistency_rate", hierarchy.get("path_consistency_rate"), level or ""],
    ]
    return rows


def _collect_coordinate_rows(results: List[Dict[str, Any]]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for result in results:
        coord_metrics = result.get("coordinate_metrics") or {}
        if not coord_metrics:
            continue
        metrics_text = ", ".join(
            f"{metric_name}={_format_value(coord_metrics.get(metric_name))}"
            for metric_name in sorted(coord_metrics)
        )
        rows.append([result.get("method"), metrics_text])
    return rows


def build_report(payload: Dict[str, Any], *, target_label_column: str | None) -> str:
    results = payload.get("results", [])
    methods = [str(item.get("method")) for item in results]
    lines = [
        "# Benchmark Report",
        "",
        "## Run Summary",
        "",
        f"- Protocol version: `{payload.get('protocol_version', '')}`",
        f"- Dataset manifest: `{payload.get('dataset_manifest', '')}`",
        f"- Methods: `{', '.join(methods)}`",
        f"- Preferred target label: `{target_label_column or 'auto'}`",
        "",
        "## Main Comparison",
        "",
        _markdown_table(
            [
                "Method",
                "Level",
                "Accuracy",
                "Macro-F1",
                "Balanced accuracy",
                "Coverage",
                "Reject rate",
                "ECE",
                "Brier",
                "AURC",
                "Unknown rate",
                "Backend",
            ],
            _collect_main_rows(results, target_label_column),
        ),
        "",
    ]
    domain_rows = _collect_domain_rows(results, target_label_column)
    if domain_rows:
        lines.extend(
            [
                "## Domain-wise Comparison",
                "",
                _markdown_table(
                    ["Method", "Domain", "Level", "Accuracy", "Macro-F1", "Coverage", "Reject rate", "Unknown rate"],
                    domain_rows,
                ),
                "",
            ]
        )
    atlas_rows = _collect_atlasmtl_rows(results, target_label_column)
    if atlas_rows:
        lines.extend(
            [
                "## atlasmtl-specific Analysis",
                "",
                _markdown_table(["Metric", "Value", "Level"], atlas_rows),
                "",
            ]
        )
    coord_rows = _collect_coordinate_rows(results)
    if coord_rows:
        lines.extend(
            [
                "## Coordinate Diagnostics",
                "",
                _markdown_table(["Method", "Metrics"], coord_rows),
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics_json).resolve()
    payload = _load_metrics(metrics_path)
    report = build_report(payload, target_label_column=args.target_label_column)
    output_path = Path(args.output).resolve() if args.output else metrics_path.with_name("benchmark_report.md")
    output_path.write_text(report, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
