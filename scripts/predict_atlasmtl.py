#!/usr/bin/env python
"""Run atlasmtl inference and write AnnData output."""

import argparse
import scanpy as sc

from atlasmtl import TrainedModel, predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adata", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = TrainedModel.load(args.model)
    adata = sc.read_h5ad(args.adata)
    result = predict(model, adata)
    adata = result.to_adata(adata)
    adata.write_h5ad(args.out)


if __name__ == "__main__":
    main()
