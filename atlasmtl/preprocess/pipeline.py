from __future__ import annotations

from anndata import AnnData

from .features import align_query_to_feature_panel, select_reference_features
from .gene_ids import canonicalize_gene_ids
from .matrix_semantics import (
    classify_count_semantics,
    detect_input_matrix_type,
    is_count_like_matrix,
    materialize_matrix,
    summarize_matrix_semantics,
)
from .metadata import attach_preprocess_metadata
from .types import FeaturePanel, PreprocessConfig, PreprocessReport


def ensure_counts_layer(
    adata: AnnData,
    config: PreprocessConfig,
) -> tuple[AnnData, dict[str, object]]:
    summary = summarize_matrix_semantics(
        adata.X,
        n_obs=config.counts_check_n_obs,
        n_vals=config.counts_check_n_vals,
        integer_tol=config.counts_check_integer_tol,
        tiny_positive_tol=config.counts_check_tiny_positive_tol,
    )
    decision = classify_count_semantics(
        summary,
        integer_fraction_threshold=config.counts_confirm_fraction,
    )
    detected_type = detect_input_matrix_type(
        adata,
        declared_type=config.input_matrix_type,
        n_obs=config.counts_check_n_obs,
        n_vals=config.counts_check_n_vals,
        integer_tol=config.counts_check_integer_tol,
        tiny_positive_tol=config.counts_check_tiny_positive_tol,
        counts_confirm_fraction=config.counts_confirm_fraction,
    )
    counts_layer = config.counts_layer
    if counts_layer in adata.layers:
        counts_ok = is_count_like_matrix(
            adata.layers[counts_layer],
            n_obs=config.counts_check_n_obs,
            n_vals=config.counts_check_n_vals,
            integer_tol=config.counts_check_integer_tol,
            tiny_positive_tol=config.counts_check_tiny_positive_tol,
            counts_confirm_fraction=config.counts_confirm_fraction,
        )
        if not counts_ok:
            raise ValueError(f"counts layer exists but is not count-like: adata.layers['{counts_layer}']")
        return adata, {
            "input_matrix_type_detected": detected_type,
            "counts_decision": decision,
            "counts_detection_summary": summary.to_dict(),
            "counts_available": True,
            "counts_source_original": f"layers/{counts_layer}",
            "counts_layer_used": counts_layer,
            "counts_layer_materialized": False,
            "counts_check_passed": True,
        }

    if decision == "counts_confirmed":
        adata.layers[counts_layer] = materialize_matrix(adata.X)
        return adata, {
            "input_matrix_type_detected": detected_type,
            "counts_decision": decision,
            "counts_detection_summary": summary.to_dict(),
            "counts_available": True,
            "counts_source_original": "X",
            "counts_layer_used": counts_layer,
            "counts_layer_materialized": True,
            "counts_check_passed": True,
        }

    if config.counts_required:
        raise ValueError(
            f"adata.X is not count-like and raw counts must be provided in adata.layers['{counts_layer}']"
        )
    return adata, {
        "input_matrix_type_detected": detected_type,
        "counts_decision": decision,
        "counts_detection_summary": summary.to_dict(),
        "counts_available": False,
        "counts_source_original": "X",
        "counts_layer_used": None,
        "counts_layer_materialized": False,
        "counts_check_passed": False,
    }


def preprocess_reference(
    adata: AnnData,
    config: PreprocessConfig,
) -> tuple[AnnData, FeaturePanel, PreprocessReport]:
    with_counts, counts_meta = ensure_counts_layer(adata.copy(), config)
    canonical, canonical_report = canonicalize_gene_ids(with_counts, config)
    selected, panel, feature_report = select_reference_features(canonical, config)
    report = PreprocessReport(
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
        ensembl_versions_stripped=canonical_report.ensembl_versions_stripped,
        gene_id_case=canonical_report.gene_id_case,
        canonical_source_column=canonical_report.canonical_source_column,
        n_genes_dropped_by_unmapped_policy=canonical_report.n_genes_dropped_by_unmapped_policy,
        n_duplicate_groups=canonical_report.n_duplicate_groups,
        duplicate_resolution_applied=canonical_report.duplicate_resolution_applied,
        preprocess_warnings=list(canonical_report.preprocess_warnings),
    )
    attach_preprocess_metadata(selected, config=config, report=report, feature_panel=panel)
    return selected, panel, report


def preprocess_query(
    adata: AnnData,
    feature_panel: FeaturePanel,
    config: PreprocessConfig,
) -> tuple[AnnData, PreprocessReport]:
    with_counts, counts_meta = ensure_counts_layer(adata.copy(), config)
    canonical, canonical_report = canonicalize_gene_ids(with_counts, config)
    aligned, align_report = align_query_to_feature_panel(canonical, feature_panel, config)
    report = PreprocessReport(
        n_input_genes=canonical_report.n_input_genes,
        n_canonical_genes=canonical_report.n_canonical_genes,
        n_duplicate_genes=canonical_report.n_duplicate_genes,
        n_unmapped_genes=canonical_report.n_unmapped_genes,
        n_features_selected=align_report.n_features_selected,
        feature_space=feature_panel.feature_space,
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
        hvg_layer_used=feature_panel.hvg_layer_used,
        matched_feature_genes=align_report.matched_feature_genes,
        missing_feature_genes=align_report.missing_feature_genes,
        ensembl_versions_stripped=canonical_report.ensembl_versions_stripped,
        gene_id_case=canonical_report.gene_id_case,
        canonical_source_column=canonical_report.canonical_source_column,
        n_genes_dropped_by_unmapped_policy=canonical_report.n_genes_dropped_by_unmapped_policy,
        n_duplicate_groups=canonical_report.n_duplicate_groups,
        duplicate_resolution_applied=canonical_report.duplicate_resolution_applied,
        preprocess_warnings=list(canonical_report.preprocess_warnings),
    )
    attach_preprocess_metadata(aligned, config=config, report=report, feature_panel=feature_panel)
    return aligned, report


def feature_panel_from_model(model) -> FeaturePanel:
    preprocess_meta = {}
    if isinstance(getattr(model, "train_config", None), dict):
        preprocess_meta = dict(model.train_config.get("preprocess") or {})
    panel_payload = preprocess_meta.get("feature_panel") if isinstance(preprocess_meta, dict) else None
    if isinstance(panel_payload, dict):
        return FeaturePanel.from_dict(panel_payload)
    preprocess_config = preprocess_meta.get("config") if isinstance(preprocess_meta, dict) else {}
    preprocess_species = str(preprocess_config.get("species", "human"))
    preprocess_var_names_type = str(preprocess_config.get("var_names_type", "ensembl"))
    preprocess_gene_id_table = preprocess_config.get("gene_id_table")
    feature_space = str(preprocess_config.get("feature_space", "whole"))
    gene_symbols = list(model.train_genes)
    return FeaturePanel(
        gene_ids=list(model.train_genes),
        gene_symbols=gene_symbols,
        feature_space=feature_space,
        n_features=len(model.train_genes),
        species=preprocess_species,
        var_names_type_original=preprocess_var_names_type,
        gene_id_table=preprocess_gene_id_table,
        hvg_method=preprocess_config.get("hvg_method"),
        n_top_genes=preprocess_config.get("n_top_genes"),
        hvg_batch_key=preprocess_config.get("hvg_batch_key"),
        counts_layer=preprocess_config.get("counts_layer"),
        hvg_layer_used=preprocess_config.get("counts_layer") if preprocess_config.get("feature_space") == "hvg" else None,
    )
