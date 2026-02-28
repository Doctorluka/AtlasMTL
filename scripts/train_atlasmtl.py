#!/usr/bin/env python
"""Train atlasmtl model from an h5ad reference dataset."""

import argparse
from anndata import read_h5ad

from atlasmtl import build_model


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
        "--no-progress",
        action="store_true",
        help="Disable the training progress bar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    adata = read_h5ad(args.adata)
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


if __name__ == "__main__":
    main()
