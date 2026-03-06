#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import anndata as ad
import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets-config",
        default="documents/experiments/2026-03-06_scanvi_param_lock_benchmark/configs/datasets.yaml",
    )
    parser.add_argument(
        "--grid-config",
        default="documents/experiments/2026-03-06_scanvi_param_lock_benchmark/configs/param_grid.yaml",
    )
    parser.add_argument(
        "--output-root",
        default="/tmp/atlasmtl_benchmarks/2026-03-06/reference_heldout",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid yaml mapping: {path}")
    return payload


def _materialize_one_dataset(dataset: Dict, *, out_root: Path, train_size: int, seed: int) -> Dict:
    dataset_name = str(dataset["dataset_name"])
    src_ref = Path(str(dataset["source_reference_h5ad"])).resolve()
    src_q5 = Path(str(dataset["source_query_5k_h5ad"])).resolve()
    src_q10 = Path(str(dataset["source_query_10k_h5ad"])).resolve()
    if not src_ref.exists():
        raise FileNotFoundError(f"missing source reference for {dataset_name}: {src_ref}")
    if not src_q5.exists():
        raise FileNotFoundError(f"missing source query 5k for {dataset_name}: {src_q5}")
    if not src_q10.exists():
        raise FileNotFoundError(f"missing source query 10k for {dataset_name}: {src_q10}")

    out_dir = out_root / dataset_name / "prepared" / "param_lock_train10k"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = ad.read_h5ad(str(src_ref))
    q5 = ad.read_h5ad(str(src_q5))
    q10 = ad.read_h5ad(str(src_q10))

    if train_size > ref.n_obs:
        raise ValueError(
            f"train_size={train_size} exceeds available reference cells={ref.n_obs} for {dataset_name}"
        )

    rng = np.random.default_rng(seed)
    selected = rng.choice(ref.n_obs, size=train_size, replace=False)
    ref_10k = ref[selected].copy()

    ref_out = out_dir / "reference_train_10k.h5ad"
    q5_out = out_dir / "heldout_test_5k.h5ad"
    q10_out = out_dir / "heldout_test_10k.h5ad"
    ref_10k.write_h5ad(str(ref_out))
    q5.write_h5ad(str(q5_out))
    q10.write_h5ad(str(q10_out))

    summary = {
        "dataset_name": dataset_name,
        "source_reference_h5ad": str(src_ref),
        "source_query_5k_h5ad": str(src_q5),
        "source_query_10k_h5ad": str(src_q10),
        "output_reference_h5ad": str(ref_out),
        "output_query_5k_h5ad": str(q5_out),
        "output_query_10k_h5ad": str(q10_out),
        "reference_n_obs_source": int(ref.n_obs),
        "reference_n_obs_output": int(ref_10k.n_obs),
        "query_5k_n_obs": int(q5.n_obs),
        "query_10k_n_obs": int(q10.n_obs),
        "n_vars_reference": int(ref_10k.n_vars),
        "n_vars_query_5k": int(q5.n_vars),
        "n_vars_query_10k": int(q10.n_vars),
        "seed": int(seed),
        "train_size": int(train_size),
    }
    (out_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    datasets_cfg = _load_yaml(Path(args.datasets_config).resolve())
    grid_cfg = _load_yaml(Path(args.grid_config).resolve())
    datasets = list(datasets_cfg.get("datasets") or [])
    if not datasets:
        raise ValueError("datasets config has empty `datasets` list")
    train_size = int(grid_cfg.get("train_size", 10000))

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict] = []
    for dataset in datasets:
        summaries.append(
            _materialize_one_dataset(dataset, out_root=out_root, train_size=train_size, seed=int(args.seed))
        )

    payload = {
        "output_root": str(out_root),
        "train_size": train_size,
        "seed": int(args.seed),
        "num_datasets": len(summaries),
        "datasets": summaries,
    }
    summary_path = out_root / "scanvi_param_lock_input_materialization_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
