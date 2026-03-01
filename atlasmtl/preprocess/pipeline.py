from __future__ import annotations

from anndata import AnnData

from .features import align_query_to_feature_panel, select_reference_features
from .gene_ids import canonicalize_gene_ids
from .metadata import attach_preprocess_metadata
from .types import FeaturePanel, PreprocessConfig, PreprocessReport


def preprocess_reference(
    adata: AnnData,
    config: PreprocessConfig,
) -> tuple[AnnData, FeaturePanel, PreprocessReport]:
    canonical, canonical_report = canonicalize_gene_ids(adata, config)
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
        ensembl_versions_stripped=canonical_report.ensembl_versions_stripped,
    )
    attach_preprocess_metadata(selected, config=config, report=report, feature_panel=panel)
    return selected, panel, report


def preprocess_query(
    adata: AnnData,
    feature_panel: FeaturePanel,
    config: PreprocessConfig,
) -> tuple[AnnData, PreprocessReport]:
    canonical, canonical_report = canonicalize_gene_ids(adata, config)
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
        matched_feature_genes=align_report.matched_feature_genes,
        missing_feature_genes=align_report.missing_feature_genes,
        ensembl_versions_stripped=canonical_report.ensembl_versions_stripped,
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
    )
