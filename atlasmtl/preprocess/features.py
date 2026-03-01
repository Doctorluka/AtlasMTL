from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse

from .types import FeaturePanel, PreprocessConfig, PreprocessReport


def select_reference_features(
    adata: AnnData,
    config: PreprocessConfig,
) -> tuple[AnnData, FeaturePanel, PreprocessReport]:
    if config.feature_space == "whole":
        selected = adata.copy()
        panel = FeaturePanel(
            gene_ids=list(selected.var_names.astype(str)),
            gene_symbols=list(selected.var.get(config.gene_symbol_column, pd.Series(selected.var_names, index=selected.var_names)).astype(str)),
            feature_space="whole",
            n_features=int(selected.n_vars),
            species=config.species,
            var_names_type_original=config.var_names_type,
            gene_id_table=config.gene_id_table,
            counts_layer=config.counts_layer,
            hvg_layer_used=None,
        )
        report = PreprocessReport(
            n_input_genes=int(adata.n_vars),
            n_canonical_genes=int(adata.n_vars),
            n_duplicate_genes=0,
            n_unmapped_genes=0,
            n_features_selected=int(selected.n_vars),
            feature_space="whole",
            species=config.species,
            var_names_type=config.var_names_type,
            mapping_resource=config.gene_id_table,
            duplicate_policy=config.duplicate_policy,
            unmapped_policy=config.unmapped_policy,
            input_matrix_type_declared=config.input_matrix_type,
            counts_available=config.counts_layer in adata.layers,
            counts_layer_used=config.counts_layer if config.counts_layer in adata.layers else None,
            counts_check_passed=config.counts_layer in adata.layers,
            hvg_layer_used=None,
        )
        return selected, panel, report

    if config.hvg_method != "seurat_v3":
        raise ValueError("only hvg_method='seurat_v3' is supported in the current preprocessing layer")
    hvg_layer_used = config.counts_layer if config.hvg_input_layer in {"auto", "counts"} else None
    if config.hvg_batch_key is not None and config.hvg_batch_key not in adata.obs.columns:
        raise ValueError(f"hvg_batch_key not found in adata.obs: {config.hvg_batch_key}")
    if hvg_layer_used is not None and hvg_layer_used not in adata.layers:
        raise ValueError(f"HVG layer not found in adata.layers: {hvg_layer_used}")

    hvg_df = sc.pp.highly_variable_genes(
        adata,
        layer=hvg_layer_used,
        n_top_genes=int(config.n_top_genes),
        flavor=config.hvg_method,
        batch_key=config.hvg_batch_key,
        inplace=False,
    )
    if "highly_variable" not in hvg_df.columns:
        raise ValueError("scanpy did not return highly_variable column")
    mask = hvg_df["highly_variable"].to_numpy(dtype=bool)
    selected = adata[:, mask].copy()
    panel = FeaturePanel(
        gene_ids=list(selected.var_names.astype(str)),
        gene_symbols=list(selected.var.get(config.gene_symbol_column, pd.Series(selected.var_names, index=selected.var_names)).astype(str)),
        feature_space="hvg",
        n_features=int(selected.n_vars),
        species=config.species,
        var_names_type_original=config.var_names_type,
        gene_id_table=config.gene_id_table,
        hvg_method=config.hvg_method,
        n_top_genes=int(config.n_top_genes),
        hvg_batch_key=config.hvg_batch_key,
        counts_layer=config.counts_layer,
        hvg_layer_used=hvg_layer_used,
    )
    report = PreprocessReport(
        n_input_genes=int(adata.n_vars),
        n_canonical_genes=int(adata.n_vars),
        n_duplicate_genes=0,
        n_unmapped_genes=0,
        n_features_selected=int(selected.n_vars),
        feature_space="hvg",
        species=config.species,
        var_names_type=config.var_names_type,
        mapping_resource=config.gene_id_table,
        duplicate_policy=config.duplicate_policy,
        unmapped_policy=config.unmapped_policy,
        input_matrix_type_declared=config.input_matrix_type,
        counts_available=config.counts_layer in adata.layers,
        counts_layer_used=config.counts_layer if config.counts_layer in adata.layers else None,
        counts_check_passed=config.counts_layer in adata.layers,
        hvg_layer_used=hvg_layer_used,
    )
    return selected, panel, report


def align_query_to_feature_panel(
    adata: AnnData,
    feature_panel: FeaturePanel,
    config: PreprocessConfig,
) -> tuple[AnnData, PreprocessReport]:
    gene_index = {gene: i for i, gene in enumerate(adata.var_names.astype(str))}
    matched = np.array([gene_index.get(gene, -1) for gene in feature_panel.gene_ids], dtype=int)
    matched_mask = matched >= 0

    n_obs = adata.n_obs
    n_features = len(feature_panel.gene_ids)
    if sparse.issparse(adata.X):
        x_new = sparse.csr_matrix((n_obs, n_features), dtype=adata.X.dtype)
        if matched_mask.any():
            x_new[:, matched_mask] = adata.X[:, matched[matched_mask]]
    else:
        x_new = np.zeros((n_obs, n_features), dtype=np.asarray(adata.X).dtype)
        if matched_mask.any():
            x_new[:, matched_mask] = np.asarray(adata.X)[:, matched[matched_mask]]

    layers_new = {}
    for name in adata.layers.keys():
        layer = adata.layers[name]
        if sparse.issparse(layer):
            layer_new = sparse.csr_matrix((n_obs, n_features), dtype=layer.dtype)
            if matched_mask.any():
                layer_new[:, matched_mask] = layer[:, matched[matched_mask]]
        else:
            layer_new = np.zeros((n_obs, n_features), dtype=np.asarray(layer).dtype)
            if matched_mask.any():
                layer_new[:, matched_mask] = np.asarray(layer)[:, matched[matched_mask]]
        layers_new[name] = layer_new

    var = pd.DataFrame(index=pd.Index(feature_panel.gene_ids, dtype=str))
    var[config.canonical_gene_id_column] = feature_panel.gene_ids
    if feature_panel.gene_symbols:
        var[config.gene_symbol_column] = feature_panel.gene_symbols
    else:
        var[config.gene_symbol_column] = feature_panel.gene_ids
    var["atlasmtl_feature_present_in_query"] = matched_mask

    aligned = AnnData(
        X=x_new,
        obs=adata.obs.copy(),
        var=var,
        obsm={key: value.copy() for key, value in adata.obsm.items()},
        uns=dict(adata.uns),
        layers=layers_new,
    )
    aligned.var_names = pd.Index(feature_panel.gene_ids, dtype=str)

    report = PreprocessReport(
        n_input_genes=int(adata.n_vars),
        n_canonical_genes=int(adata.n_vars),
        n_duplicate_genes=0,
        n_unmapped_genes=0,
        n_features_selected=int(aligned.n_vars),
        feature_space=feature_panel.feature_space,
        species=config.species,
        var_names_type=config.var_names_type,
        mapping_resource=config.gene_id_table,
        duplicate_policy=config.duplicate_policy,
        unmapped_policy=config.unmapped_policy,
        input_matrix_type_declared=config.input_matrix_type,
        counts_available=config.counts_layer in adata.layers,
        counts_layer_used=config.counts_layer if config.counts_layer in adata.layers else None,
        counts_check_passed=config.counts_layer in adata.layers,
        hvg_layer_used=feature_panel.hvg_layer_used,
        matched_feature_genes=int(matched_mask.sum()),
        missing_feature_genes=int((~matched_mask).sum()),
    )
    return aligned, report
