from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np


def _str2bool(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-h5ad", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--trainer-backend", choices=["wrapped_logreg", "formal"], default="formal")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--feature-selection", type=_str2bool, default=True)
    parser.add_argument("--balance-cell-type", type=_str2bool, default=True)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--top-genes", type=int, default=500)
    parser.add_argument("--use-gpu", type=_str2bool, default=False)
    parser.add_argument("--with-mean", type=_str2bool, default=False)
    parser.add_argument("--min-cells-per-class", type=int, default=0)
    parser.add_argument("--summary-json")
    return parser.parse_args()


def _filter_min_cells(adata: ad.AnnData, label_column: str, min_cells_per_class: int) -> tuple[ad.AnnData, int]:
    if min_cells_per_class <= 0:
        return adata, 0
    counts = adata.obs[label_column].astype(str).value_counts()
    keep = counts[counts >= min_cells_per_class].index
    filtered = adata[adata.obs[label_column].astype(str).isin(keep)].copy()
    return filtered, int(adata.n_obs - filtered.n_obs)


def _train_wrapped_logreg(adata: ad.AnnData, args: argparse.Namespace):
    from celltypist import models
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    classifier = LogisticRegression(
        max_iter=int(args.max_iter),
        solver="saga",
        n_jobs=int(args.n_jobs),
    )
    classifier.fit(adata.X, adata.obs[args.label_column].astype(str).to_numpy())
    classifier.features = np.asarray(adata.var_names.astype(str), dtype=object)

    scaler = StandardScaler(with_mean=bool(args.with_mean), with_std=True)
    scaler.fit(adata.X)

    return models.Model(
        classifier,
        scaler,
        {
            "date": "2026-03-04",
            "details": f"wrapped-logreg celltypist comparator for {args.label_column}",
            "source": str(Path(args.reference_h5ad).resolve()),
            "version": "2026-03-04",
        },
    )


def _train_formal_celltypist(adata: ad.AnnData, args: argparse.Namespace):
    import importlib

    import celltypist
    celltypist_train_module = importlib.import_module("celltypist.train")

    train_kwargs = dict(
        labels=args.label_column,
        n_jobs=int(args.n_jobs),
        feature_selection=bool(args.feature_selection),
        balance_cell_type=bool(args.balance_cell_type),
        use_GPU=bool(args.use_gpu),
        batch_size=int(args.batch_size),
        top_genes=int(args.top_genes),
        max_iter=int(args.max_iter),
        with_mean=bool(args.with_mean),
        date="2026-03-04",
        details=f"formal celltypist comparator for {args.label_column}",
        source=str(Path(args.reference_h5ad).resolve()),
        version="2026-03-04",
    )

    try:
        return celltypist.train(adata, **train_kwargs)
    except TypeError as exc:
        if "multi_class" not in str(exc):
            raise
        from benchmark.methods.celltypist_compat import CompatibleCellTypistLogisticRegression

        celltypist_train_module.LogisticRegression = CompatibleCellTypistLogisticRegression
        return celltypist.train(adata, **train_kwargs)


def main() -> None:
    args = parse_args()
    adata = ad.read_h5ad(args.reference_h5ad)
    if args.label_column not in adata.obs.columns:
        raise ValueError(f"missing label column: {args.label_column}")

    adata, n_filtered = _filter_min_cells(adata, args.label_column, int(args.min_cells_per_class))
    if adata.n_obs == 0:
        raise ValueError("no cells remain after min-cells-per-class filtering")

    start = time.perf_counter()
    if args.trainer_backend == "wrapped_logreg":
        model = _train_wrapped_logreg(adata, args)
    else:
        model = _train_formal_celltypist(adata, args)
    elapsed = round(time.perf_counter() - start, 4)

    output_path = Path(args.output_model).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.write(output_path)

    summary = {
        "trainer_backend": args.trainer_backend,
        "reference_h5ad": str(Path(args.reference_h5ad).resolve()),
        "label_column": args.label_column,
        "n_obs_used": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "n_labels": int(adata.obs[args.label_column].astype(str).nunique()),
        "n_filtered_cells": int(n_filtered),
        "elapsed_seconds": elapsed,
        "output_model": str(output_path),
        "trainer_config": {
            "max_iter": int(args.max_iter),
            "n_jobs": int(args.n_jobs),
            "feature_selection": bool(args.feature_selection),
            "balance_cell_type": bool(args.balance_cell_type),
            "batch_size": int(args.batch_size),
            "top_genes": int(args.top_genes),
            "use_gpu": bool(args.use_gpu),
            "with_mean": bool(args.with_mean),
            "min_cells_per_class": int(args.min_cells_per_class),
        },
    }
    if args.summary_json:
        summary_path = Path(args.summary_json).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
