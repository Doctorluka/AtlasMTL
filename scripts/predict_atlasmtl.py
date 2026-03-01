#!/usr/bin/env python
"""Run atlasmtl inference and write AnnData output."""

import argparse
import json
from pathlib import Path
from anndata import read_h5ad
import numpy as np

from atlasmtl import TrainedModel, predict
from atlasmtl.preprocess import PreprocessConfig, feature_panel_from_model, preprocess_query


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
        "--var-names-type",
        choices=["symbol", "ensembl"],
        default=None,
        help="Optional preprocessing input gene namespace for query data. Requires --species when set.",
    )
    parser.add_argument(
        "--species",
        choices=["human", "mouse", "rat"],
        default=None,
        help="Optional preprocessing species for query data. Requires --var-names-type when set.",
    )
    parser.add_argument(
        "--gene-id-table",
        default=None,
        help="Optional path to a gene-id mapping table. Defaults to the packaged atlasmtl resource.",
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
        help="Disable the inference progress bar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = TrainedModel.load(args.model)
    adata = read_h5ad(args.adata)
    preprocess_payload = None
    if args.var_names_type is not None or args.species is not None:
        if args.var_names_type is None or args.species is None:
            raise ValueError("Specify both --var-names-type and --species to enable preprocessing.")
        train_preprocess = {}
        if isinstance(model.train_config, dict):
            train_preprocess = dict(model.train_config.get("preprocess") or {})
        preprocess_config = PreprocessConfig(
            var_names_type=args.var_names_type,
            species=args.species,
            gene_id_table=args.gene_id_table or (train_preprocess.get("config") or {}).get("gene_id_table"),
            feature_space=str((train_preprocess.get("config") or {}).get("feature_space", "whole")),
            n_top_genes=int((train_preprocess.get("config") or {}).get("n_top_genes", 3000)),
            hvg_method=str((train_preprocess.get("config") or {}).get("hvg_method", "seurat_v3")),
            hvg_batch_key=(train_preprocess.get("config") or {}).get("hvg_batch_key"),
            duplicate_policy=args.duplicate_policy,
            unmapped_policy=args.unmapped_policy,
        )
        feature_panel = feature_panel_from_model(model)
        adata, preprocess_report = preprocess_query(adata, feature_panel, preprocess_config)
        preprocess_payload = {
            "config": preprocess_config.to_dict(),
            "feature_panel": feature_panel.to_dict(),
            "query_report": preprocess_report.to_dict(),
        }
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
    out_dir = Path(args.out).resolve().parent
    (out_dir / "predict_run_manifest.json").write_text(
        json.dumps(
            _json_safe(
                {
                "schema_version": 1,
                "command": "predict_atlasmtl",
                "model": str(Path(args.model).resolve()),
                "out": str(Path(args.out).resolve()),
                "device": args.device,
                "prediction_metadata": result.metadata,
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
