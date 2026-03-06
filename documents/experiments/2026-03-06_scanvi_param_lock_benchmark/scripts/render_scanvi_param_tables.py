#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="documents/experiments/2026-03-06_scanvi_param_lock_benchmark/results_summary",
    )
    return parser.parse_args()


def _manual_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._\n"
    columns = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in df.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def _to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._\n"
    try:
        return df.to_markdown(index=False) + "\n"
    except Exception:
        return _manual_markdown_table(df)


def _render_one(title: str, csv_path: Path, md_path: Path) -> None:
    if not csv_path.exists() or csv_path.read_text(encoding="utf-8").strip() == "":
        md_path.write_text(f"# {title}\n\n_No data._\n", encoding="utf-8")
        return
    df = pd.read_csv(csv_path)
    content = f"# {title}\n\n{_to_md(df)}"
    md_path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    _render_one(
        "Scanvi Param Sweep Raw Results",
        results_dir / "sweep_raw_results.csv",
        results_dir / "sweep_raw_results.md",
    )
    _render_one(
        "Stage A Param Ranking",
        results_dir / "stage_a_param_ranking.csv",
        results_dir / "stage_a_param_ranking.md",
    )
    _render_one(
        "Stage B Stability Summary",
        results_dir / "stage_b_stability.csv",
        results_dir / "stage_b_stability.md",
    )
    print(str(results_dir))


if __name__ == "__main__":
    main()
