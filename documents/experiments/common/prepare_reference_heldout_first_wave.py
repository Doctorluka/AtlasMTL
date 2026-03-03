#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import anndata as ad
import yaml

from atlasmtl.preprocess import (
    PreprocessConfig,
    ensure_counts_layer,
    make_group_split_plan,
    materialize_group_split_subsets,
    save_feature_panel,
)
from atlasmtl.preprocess.features import align_query_to_feature_panel, select_reference_features
from atlasmtl.preprocess.gene_ids import canonicalize_gene_ids
from atlasmtl.preprocess.metadata import attach_preprocess_metadata
from atlasmtl.preprocess.types import PreprocessReport


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--source-h5ad", required=True)
    parser.add_argument("--split-key", required=True)
    parser.add_argument("--domain-key", required=True)
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--build-size", type=int, required=True)
    parser.add_argument("--predict-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-candidates", type=int, default=128)
    return parser.parse_args()


def _load_manifest(path: str) -> Dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dataset manifest must be a YAML mapping")
    return payload


def _build_preprocess_config(manifest: Dict[str, Any]) -> PreprocessConfig:
    hvg_cfg = dict(manifest.get("hvg_config") or {})
    return PreprocessConfig(
        var_names_type=str(manifest["var_names_type"]),
        species=str(manifest["species"]),
        gene_id_table=manifest.get("gene_id_table"),
        canonical_target=str(manifest.get("canonical_target", "ensembl")),
        ensembl_source_column=manifest.get("ensembl_source_column"),
        symbol_source_column=manifest.get("symbol_source_column"),
        prefer_existing_ensembl_column=bool(manifest.get("prefer_existing_ensembl_column", True)),
        mapping_table_kind=str(manifest.get("mapping_table_kind", "biomart_human_mouse_rat")),
        input_matrix_type=str(manifest.get("input_matrix_type", "infer")),
        counts_layer=str(manifest.get("counts_layer", "counts")),
        counts_required=True,
        counts_check_tiny_positive_tol=float(manifest.get("counts_check_tiny_positive_tol", 1e-8)),
        counts_confirm_fraction=float(manifest.get("counts_confirm_fraction", 0.999)),
        feature_space=str(manifest.get("feature_space", "hvg")),
        n_top_genes=int(hvg_cfg.get("n_top_genes", 3000)),
        hvg_method=str(hvg_cfg.get("method", "seurat_v3")),
        hvg_batch_key=hvg_cfg.get("batch_key"),
        duplicate_policy=str(manifest.get("duplicate_policy", "sum")),
        unmapped_policy=str(manifest.get("unmapped_policy", "drop")),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _compose_preprocess_report(
    *,
    config: PreprocessConfig,
    canonical_report: PreprocessReport,
    counts_meta: Dict[str, Any],
    feature_report: PreprocessReport,
    matched_feature_genes: int | None = None,
    missing_feature_genes: int | None = None,
) -> PreprocessReport:
    return PreprocessReport(
        n_input_genes=canonical_report.n_input_genes,
        n_canonical_genes=canonical_report.n_canonical_genes,
        n_duplicate_genes=canonical_report.n_duplicate_genes,
        n_unmapped_genes=canonical_report.n_unmapped_genes,
        n_features_selected=feature_report.n_features_selected,
        feature_space=config.feature_space,
        species=config.species,
        var_names_type=config.var_names_type,
        mapping_resource=canonical_report.mapping_resource,
        duplicate_policy=config.duplicate_policy,
        unmapped_policy=config.unmapped_policy,
        input_matrix_type_declared=config.input_matrix_type,
        input_matrix_type_detected=str(counts_meta["input_matrix_type_detected"]),
        counts_available=bool(counts_meta["counts_available"]),
        counts_layer_used=counts_meta["counts_layer_used"],
        counts_check_passed=bool(counts_meta["counts_check_passed"]),
        counts_decision=str(counts_meta["counts_decision"]),
        counts_detection_summary=dict(counts_meta["counts_detection_summary"]),
        counts_source_original=str(counts_meta["counts_source_original"]),
        counts_layer_materialized=bool(counts_meta["counts_layer_materialized"]),
        hvg_layer_used=feature_report.hvg_layer_used,
        matched_feature_genes=matched_feature_genes,
        missing_feature_genes=missing_feature_genes,
        ensembl_versions_stripped=canonical_report.ensembl_versions_stripped,
        gene_id_case=canonical_report.gene_id_case,
        canonical_source_column=canonical_report.canonical_source_column,
        n_genes_dropped_by_unmapped_policy=canonical_report.n_genes_dropped_by_unmapped_policy,
        n_duplicate_groups=canonical_report.n_duplicate_groups,
        duplicate_resolution_applied=canonical_report.duplicate_resolution_applied,
        preprocess_warnings=list(canonical_report.preprocess_warnings),
    )


def main() -> None:
    args = parse_args()
    manifest = _load_manifest(args.dataset_manifest)
    preprocess_config = _build_preprocess_config(manifest)
    source_h5ad = Path(args.source_h5ad).resolve()

    raw_adata = ad.read_h5ad(str(source_h5ad))
    with_counts, full_counts_meta = ensure_counts_layer(raw_adata, preprocess_config)
    canonical_adata, full_canonical_report = canonicalize_gene_ids(with_counts, preprocess_config)

    split_plan = make_group_split_plan(
        canonical_adata,
        split_key=args.split_key,
        target_label=args.target_label,
        build_size=args.build_size,
        predict_size=args.predict_size,
        seed=args.seed,
        n_candidates=args.n_candidates,
    )
    split_plan["domain_key"] = args.domain_key
    split_plan["dataset_name"] = manifest.get("dataset_name")
    split_plan["source_h5ad"] = str(source_h5ad)

    materialized = materialize_group_split_subsets(
        canonical_adata,
        split_plan,
        build_size=args.build_size,
        predict_size=args.predict_size,
        seed=args.seed,
    )
    reference_canonical = materialized["reference_build_adata"]
    predict_canonical = materialized["predict_adata"]
    split_summary = dict(materialized["split_summary"])
    split_summary["domain_key"] = args.domain_key
    split_summary["dataset_name"] = manifest.get("dataset_name")
    split_summary["source_h5ad"] = str(source_h5ad)

    reference_prepared, feature_panel, reference_feature_report = select_reference_features(
        reference_canonical,
        preprocess_config,
    )
    predict_prepared, predict_align_report = align_query_to_feature_panel(
        predict_canonical,
        feature_panel,
        preprocess_config,
    )

    reference_report = _compose_preprocess_report(
        config=preprocess_config,
        canonical_report=full_canonical_report,
        counts_meta=full_counts_meta,
        feature_report=reference_feature_report,
    )
    predict_report = _compose_preprocess_report(
        config=preprocess_config,
        canonical_report=full_canonical_report,
        counts_meta=full_counts_meta,
        feature_report=predict_align_report,
        matched_feature_genes=predict_align_report.matched_feature_genes,
        missing_feature_genes=predict_align_report.missing_feature_genes,
    )
    attach_preprocess_metadata(
        reference_prepared,
        config=preprocess_config,
        report=reference_report,
        feature_panel=feature_panel,
    )
    attach_preprocess_metadata(
        predict_prepared,
        config=preprocess_config,
        report=predict_report,
        feature_panel=feature_panel,
    )

    reference_path = Path(str(manifest["reference_h5ad"])).expanduser().resolve()
    query_path = Path(str(manifest["query_h5ad"])).expanduser().resolve()
    output_dir = reference_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_prepared.write_h5ad(reference_path)
    predict_prepared.write_h5ad(query_path)
    save_feature_panel(feature_panel, str(output_dir / "feature_panel.json"))
    (output_dir / "split_plan.json").write_text(json.dumps(_json_safe(split_plan), indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "split_summary.json").write_text(json.dumps(_json_safe(split_summary), indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "preprocessing_summary.json").write_text(
        json.dumps(
            {
                "dataset_name": manifest.get("dataset_name"),
                "source_h5ad": str(source_h5ad),
                "reference_output_h5ad": str(reference_path),
                "query_output_h5ad": str(query_path),
                "target_label": args.target_label,
                "split_key": args.split_key,
                "domain_key": args.domain_key,
                "build_size": int(args.build_size),
                "predict_size": int(args.predict_size),
                "preprocess_config": preprocess_config.to_dict(),
                "full_counts_meta": _json_safe(full_counts_meta),
                "full_canonical_report": full_canonical_report.to_dict(),
                "reference_report": reference_report.to_dict(),
                "predict_report": predict_report.to_dict(),
                "feature_panel": feature_panel.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "reference_h5ad": str(reference_path),
                "query_h5ad": str(query_path),
                "feature_panel_json": str(output_dir / "feature_panel.json"),
                "split_summary_json": str(output_dir / "split_summary.json"),
                "preprocessing_summary_json": str(output_dir / "preprocessing_summary.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
