#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-reference-h5ad",
        default="/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/prepared/group_split_v2_train100k_test10k_nested5k/reference_train.h5ad",
    )
    parser.add_argument(
        "--source-query-h5ad",
        default="/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/prepared/group_split_v2_train100k_test10k_nested5k/heldout_test_5k.h5ad",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/atlasmtl_benchmarks/2026-03-05/reference_heldout/HLCA_Core/prepared/formal_train10k_test5k",
    )
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = ad.read_h5ad(str(Path(args.source_reference_h5ad).resolve()))
    query = ad.read_h5ad(str(Path(args.source_query_h5ad).resolve()))

    if args.train_size > ref.n_obs:
        raise ValueError(f"train-size {args.train_size} exceeds available reference cells {ref.n_obs}")

    rng = np.random.default_rng(args.seed)
    selected_idx = rng.choice(ref.n_obs, size=args.train_size, replace=False)
    ref_10k = ref[selected_idx].copy()

    ref_out = out_dir / "reference_train_10k.h5ad"
    query_out = out_dir / "heldout_test_5k.h5ad"
    ref_10k.write_h5ad(ref_out)
    query.write_h5ad(query_out)

    payload = {
        "source_reference_h5ad": str(Path(args.source_reference_h5ad).resolve()),
        "source_query_h5ad": str(Path(args.source_query_h5ad).resolve()),
        "reference_output_h5ad": str(ref_out),
        "query_output_h5ad": str(query_out),
        "reference_n_obs_source": int(ref.n_obs),
        "query_n_obs_source": int(query.n_obs),
        "reference_n_obs_output": int(ref_10k.n_obs),
        "query_n_obs_output": int(query.n_obs),
        "seed": int(args.seed),
        "train_size": int(args.train_size),
    }
    (out_dir / "split_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
