#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import anndata as ad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-h5ad", required=True)
    parser.add_argument("--label-columns", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def build_hierarchy_rules(reference_h5ad: Path, label_columns: List[str]) -> Dict[str, Dict[str, object]]:
    adata = ad.read_h5ad(reference_h5ad)
    rules: Dict[str, Dict[str, object]] = {}
    for parent_col, child_col in zip(label_columns[:-1], label_columns[1:]):
        frame = adata.obs.loc[:, [parent_col, child_col]].dropna().copy()
        frame[parent_col] = frame[parent_col].astype(str)
        frame[child_col] = frame[child_col].astype(str)
        dedup = frame.drop_duplicates(subset=[child_col, parent_col], keep="first")
        rules[str(child_col)] = {
            "parent_col": str(parent_col),
            "child_to_parent": dedup.set_index(child_col)[parent_col].to_dict(),
        }
    return rules


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = build_hierarchy_rules(Path(args.reference_h5ad).resolve(), list(args.label_columns))
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": output_json.as_posix(),
                "edge_count": len(payload),
                "label_columns": list(args.label_columns),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
