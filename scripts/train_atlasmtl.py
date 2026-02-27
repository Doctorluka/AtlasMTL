#!/usr/bin/env python
"""Train atlasmtl model from an h5ad reference dataset."""

import argparse
import scanpy as sc

from atlasmtl import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--latent-key", default="X_ref_latent")
    parser.add_argument("--umap-key", default="X_umap")
    args = parser.parse_args()

    adata = sc.read_h5ad(args.adata)
    model = build_model(
        adata=adata,
        label_columns=args.labels,
        coord_targets={"latent": args.latent_key, "umap": args.umap_key},
        latent_source="internal_preferred",
    )
    model.save(args.out)


if __name__ == "__main__":
    main()
