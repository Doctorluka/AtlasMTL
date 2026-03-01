from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
from celltypist import models
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-h5ad", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adata = ad.read_h5ad(args.reference_h5ad)
    if args.label_column not in adata.obs.columns:
        raise ValueError(f"missing label column: {args.label_column}")

    classifier = LogisticRegression(
        max_iter=int(args.max_iter),
        solver="saga",
        n_jobs=int(args.n_jobs),
    )
    classifier.fit(adata.X, adata.obs[args.label_column].astype(str).to_numpy())
    classifier.features = np.asarray(adata.var_names.astype(str), dtype=object)

    scaler = StandardScaler(with_mean=False, with_std=True)
    scaler.fit(adata.X)

    model = models.Model(
        classifier,
        scaler,
        {
            "date": "2026-03-01",
            "details": f"atlasmtl real benchmark model for {args.label_column}",
            "source": "sampled_adata_10k preprocessed reference",
            "version": "2026-03-01",
        },
    )
    output_path = Path(args.output_model).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.write(output_path)
    print(str(output_path))


if __name__ == "__main__":
    main()
