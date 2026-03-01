from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad


LEVELS = [
    ("anno_lv2", "anno_lv1"),
    ("anno_lv3", "anno_lv2"),
    ("anno_lv4", "anno_lv3"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-h5ad", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adata = ad.read_h5ad(args.reference_h5ad)
    payload = {}
    for child_col, parent_col in LEVELS:
        frame = adata.obs.loc[:, [child_col, parent_col]].dropna()
        frame[child_col] = frame[child_col].astype(str)
        frame[parent_col] = frame[parent_col].astype(str)
        dedup = frame.drop_duplicates(subset=[child_col, parent_col], keep="first")
        child_to_parent = dedup.set_index(child_col)[parent_col].to_dict()
        payload[child_col] = {
            "parent_col": parent_col,
            "child_to_parent": child_to_parent,
        }
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()
