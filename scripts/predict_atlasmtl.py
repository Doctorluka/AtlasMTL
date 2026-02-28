#!/usr/bin/env python
"""Run atlasmtl inference and write AnnData output."""

import argparse
from anndata import read_h5ad

from atlasmtl import TrainedModel, predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run atlasmtl inference on a query .h5ad file and optionally write "
            "selected outputs back into AnnData."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to atlasmtl model artifact: model.pth or model_manifest.json.",
    )
    parser.add_argument("--adata", required=True, help="Path to the query .h5ad file.")
    parser.add_argument("--out", required=True, help="Output path for the annotated .h5ad file.")
    parser.add_argument(
        "--knn-correction",
        choices=["off", "low_conf_only", "all"],
        default="low_conf_only",
        help="KNN usage mode during prediction. Default: low_conf_only.",
    )
    parser.add_argument(
        "--confidence-high",
        type=float,
        default=0.7,
        help="High confidence threshold used for low-confidence gating. Default: 0.7.",
    )
    parser.add_argument(
        "--confidence-low",
        type=float,
        default=0.4,
        help="Low confidence threshold used for Unknown assignment. Default: 0.4.",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.2,
        help="Minimum top1-top2 probability margin for a confident prediction. Default: 0.2.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=15,
        help="Number of reference neighbors used for KNN voting. Default: 15.",
    )
    parser.add_argument(
        "--knn-conf-low",
        type=float,
        default=0.6,
        help="Minimum KNN vote fraction required to avoid Unknown in the closed loop. Default: 0.6.",
    )
    parser.add_argument(
        "--input-transform",
        choices=["binary", "float"],
        default=None,
        help="Optional override for the model's stored input transform.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Inference batch size. Default: 256.",
    )
    parser.add_argument(
        "--write-mode",
        choices=["minimal", "standard", "full"],
        default="standard",
        help="How much prediction information to write back into AnnData. Default: standard.",
    )
    parser.add_argument(
        "--include-coords",
        action="store_true",
        help="Write predicted coordinates into obsm['X_pred_*']. Disabled by default.",
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Do not write uns['atlasmtl'] metadata into the output AnnData.",
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
        help="Disable the inference progress bar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = TrainedModel.load(args.model)
    adata = read_h5ad(args.adata)
    result = predict(
        model,
        adata,
        knn_correction=args.knn_correction,
        confidence_high=args.confidence_high,
        confidence_low=args.confidence_low,
        margin_threshold=args.margin_threshold,
        knn_k=args.knn_k,
        knn_conf_low=args.knn_conf_low,
        input_transform=args.input_transform,
        batch_size=args.batch_size,
        num_threads=args.num_threads if args.num_threads == "max" else int(args.num_threads),
        device=args.device,
        show_progress=not args.no_progress,
    )
    adata = result.to_adata(
        adata,
        mode=args.write_mode,
        include_coords=args.include_coords,
        include_metadata=not args.skip_metadata,
    )
    adata.write_h5ad(args.out)


if __name__ == "__main__":
    main()
