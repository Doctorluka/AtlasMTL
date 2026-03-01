#!/usr/bin/env python
"""Train atlasmtl model from an h5ad reference dataset."""

import argparse
import json
from pathlib import Path
from anndata import read_h5ad
import numpy as np

from atlasmtl import build_model
from atlasmtl.preprocess import PreprocessConfig, preprocess_reference


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an atlasmtl reference model from a reference .h5ad file. "
            "The input AnnData must provide label columns in obs and coordinate "
            "targets in obsm."
        )
    )
    parser.add_argument("--adata", required=True, help="Path to the reference .h5ad file.")
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for the main model artifact, typically ending in .pth.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Ordered label columns in adata.obs. Each column becomes one prediction head.",
    )
    parser.add_argument(
        "--latent-key",
        default=None,
        help="Optional key in adata.obsm used as the latent coordinate regression target.",
    )
    parser.add_argument(
        "--umap-key",
        default=None,
        help="Optional key in adata.obsm used as the UMAP coordinate regression target.",
    )
    parser.add_argument(
        "--no-coords",
        action="store_true",
        help="Disable coordinate heads and train a phmap-style label-only model.",
    )
    parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Shared encoder hidden sizes, for example: --hidden-sizes 256 128.",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.3,
        help="Dropout rate used in the shared encoder. Default: 0.3.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size. Default: 256.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=40,
        help="Maximum number of training epochs. Default: 40.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate. Default: 1e-3.",
    )
    parser.add_argument(
        "--input-transform",
        choices=["binary", "float"],
        default="binary",
        help="Input preprocessing for adata.X. 'binary' is the recommended default.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="Optional validation split fraction. Set > 0 to enable validation.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Validation epochs without improvement allowed before early stopping. Disabled by default.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation-loss improvement required to reset early stopping.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for the optional validation split. Default: 42.",
    )
    parser.add_argument(
        "--reference-storage",
        choices=["external", "full"],
        default="external",
        help="How to store KNN reference data with the model. 'external' is recommended.",
    )
    parser.add_argument(
        "--reference-path",
        default=None,
        help="Optional custom path for external reference storage.",
    )
    parser.add_argument(
        "--num-threads",
        default="10",
        help="PyTorch CPU threads to use. Default: 10. Use 'max' for up to 80%% of CPUs.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device. Default: auto.",
    )
    parser.add_argument(
        "--var-names-type",
        choices=["symbol", "ensembl"],
        default=None,
        help="Optional preprocessing input gene namespace. Requires --species when set.",
    )
    parser.add_argument(
        "--species",
        choices=["human", "mouse", "rat"],
        default=None,
        help="Optional preprocessing species. Requires --var-names-type when set.",
    )
    parser.add_argument(
        "--gene-id-table",
        default=None,
        help="Optional path to a gene-id mapping table. Defaults to the packaged atlasmtl resource.",
    )
    parser.add_argument(
        "--feature-space",
        choices=["hvg", "whole"],
        default="hvg",
        help="Feature selection mode used during preprocessing. Default: hvg.",
    )
    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=3000,
        help="Number of HVGs to keep when --feature-space hvg. Default: 3000.",
    )
    parser.add_argument(
        "--hvg-method",
        choices=["seurat_v3"],
        default="seurat_v3",
        help="HVG method used during preprocessing. Default: seurat_v3.",
    )
    parser.add_argument(
        "--hvg-batch-key",
        default=None,
        help="Optional obs column used as batch_key during HVG selection.",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=["sum", "mean", "first", "error"],
        default="sum",
        help="How to aggregate duplicate canonical genes during preprocessing. Default: sum.",
    )
    parser.add_argument(
        "--unmapped-policy",
        choices=["drop", "keep_original", "error"],
        default="drop",
        help="How to handle unmapped genes during preprocessing. Default: drop.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the training progress bar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    adata = read_h5ad(args.adata)
    preprocess_payload = None
    if args.var_names_type is not None or args.species is not None:
        if args.var_names_type is None or args.species is None:
            raise ValueError("Specify both --var-names-type and --species to enable preprocessing.")
        preprocess_config = PreprocessConfig(
            var_names_type=args.var_names_type,
            species=args.species,
            gene_id_table=args.gene_id_table,
            feature_space=args.feature_space,
            n_top_genes=args.n_top_genes,
            hvg_method=args.hvg_method,
            hvg_batch_key=args.hvg_batch_key,
            duplicate_policy=args.duplicate_policy,
            unmapped_policy=args.unmapped_policy,
        )
        adata, feature_panel, preprocess_report = preprocess_reference(adata, preprocess_config)
        preprocess_payload = {
            "config": preprocess_config.to_dict(),
            "feature_panel": feature_panel.to_dict(),
            "reference_report": preprocess_report.to_dict(),
        }
    if args.no_coords:
        coord_targets = {}
    elif args.latent_key is None and args.umap_key is None:
        coord_targets = None
    else:
        if args.latent_key is None or args.umap_key is None:
            raise ValueError("Specify both --latent-key and --umap-key, or neither, or use --no-coords.")
        coord_targets = {"latent": args.latent_key, "umap": args.umap_key}
    model = build_model(
        adata=adata,
        label_columns=args.labels,
        coord_targets=coord_targets,
        latent_source="internal_preferred",
        hidden_sizes=args.hidden_sizes,
        dropout_rate=args.dropout_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        input_transform=args.input_transform,
        val_fraction=args.val_fraction,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        random_state=args.random_state,
        reference_storage=args.reference_storage,
        reference_path=args.reference_path,
        num_threads=args.num_threads if args.num_threads == "max" else int(args.num_threads),
        device=args.device,
        show_progress=not args.no_progress,
    )
    model.save(args.out)
    out_dir = Path(args.out).resolve().parent
    (out_dir / "train_run_manifest.json").write_text(
        json.dumps(
            _json_safe(
                {
                "schema_version": 1,
                "command": "train_atlasmtl",
                "out": str(Path(args.out).resolve()),
                "labels": list(args.labels),
                "device": args.device,
                "train_config": model.train_config,
                "preprocess": preprocess_payload,
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
