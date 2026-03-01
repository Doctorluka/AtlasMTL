from __future__ import annotations

import gzip
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from ..resources import get_resource_path
from .types import PreprocessConfig, PreprocessReport


SPECIES_COLUMNS = {
    "human": ("human_ensembl_gene_id", "human_gene_symbol"),
    "mouse": ("mouse_ensembl_gene_id", "mouse_gene_symbol"),
    "rat": ("rat_ensembl_gene_id", "rat_gene_symbol"),
}


def default_gene_id_table_path() -> str:
    return str(get_resource_path("gene_id", "biomart_human_mouse_rat.tsv.gz"))


def strip_ensembl_version(value: str) -> str:
    if "." in value:
        return value.split(".", 1)[0]
    return value


def load_gene_id_table(path: Optional[str] = None) -> pd.DataFrame:
    resolved = Path(path or default_gene_id_table_path())
    opener = gzip.open if resolved.suffix == ".gz" else open
    with opener(resolved, "rt", encoding="utf-8") as handle:
        return pd.read_csv(handle, sep="\t")


def _canonicalize_ids(raw_names: pd.Index, config: PreprocessConfig) -> tuple[pd.Series, pd.Series, int]:
    raw = raw_names.astype(str)
    stripped_count = 0
    if config.var_names_type == "ensembl":
        canonical = pd.Series(raw, index=np.arange(len(raw)), dtype="object")
        if config.strip_ensembl_version:
            stripped = canonical.map(strip_ensembl_version)
            stripped_count = int((canonical != stripped).sum())
            canonical = stripped
        table = load_gene_id_table(config.gene_id_table)
        ensembl_col, symbol_col = SPECIES_COLUMNS[config.species]
        lookup = (
            table[[ensembl_col, symbol_col]]
            .dropna(subset=[ensembl_col])
            .assign(**{ensembl_col: lambda df: df[ensembl_col].astype(str).map(strip_ensembl_version)})
            .drop_duplicates(subset=[ensembl_col], keep="first")
            .set_index(ensembl_col)[symbol_col]
        )
        gene_symbol = canonical.map(lookup).fillna(pd.Series(raw, index=canonical.index))
        return canonical, gene_symbol, stripped_count

    table = load_gene_id_table(config.gene_id_table)
    ensembl_col, symbol_col = SPECIES_COLUMNS[config.species]
    mapping = (
        table[[symbol_col, ensembl_col]]
        .dropna(subset=[symbol_col, ensembl_col])
        .assign(
            **{
                symbol_col: lambda df: df[symbol_col].astype(str),
                ensembl_col: lambda df: df[ensembl_col].astype(str).map(strip_ensembl_version),
            }
        )
        .drop_duplicates(subset=[symbol_col], keep="first")
        .set_index(symbol_col)[ensembl_col]
    )
    gene_symbol = pd.Series(raw, index=np.arange(len(raw)), dtype="object")
    canonical = gene_symbol.map(mapping)
    return canonical, gene_symbol, stripped_count


def _aggregate_matrix(matrix, groups: list[np.ndarray], policy: str):
    if sparse.issparse(matrix):
        csc = matrix.tocsc()
        cols = []
        for idx in groups:
            if len(idx) == 1 or policy == "first":
                cols.append(csc[:, idx[0]])
                continue
            block = csc[:, idx]
            agg = block.sum(axis=1)
            if policy == "mean":
                agg = agg / float(len(idx))
            cols.append(sparse.csc_matrix(agg))
        return sparse.hstack(cols, format="csr")

    arr = np.asarray(matrix)
    cols_out = []
    for idx in groups:
        if len(idx) == 1 or policy == "first":
            cols_out.append(arr[:, idx[0]])
            continue
        block = arr[:, idx]
        if policy == "sum":
            cols_out.append(block.sum(axis=1))
        elif policy == "mean":
            cols_out.append(block.mean(axis=1))
        else:
            cols_out.append(block[:, 0])
    return np.column_stack(cols_out).astype(arr.dtype, copy=False)


def _aggregate_layers(layers: dict[str, object], groups: list[np.ndarray], policy: str) -> dict[str, object]:
    return {name: _aggregate_matrix(layer, groups, policy) for name, layer in layers.items()}


def canonicalize_gene_ids(
    adata: AnnData,
    config: PreprocessConfig,
) -> tuple[AnnData, PreprocessReport]:
    canonical_ids, gene_symbols, stripped_count = _canonicalize_ids(adata.var_names, config)
    n_input = int(adata.n_vars)
    unmapped_mask = canonical_ids.isna() | (canonical_ids.astype(str).str.len() == 0)
    n_unmapped = int(unmapped_mask.sum())
    if n_unmapped and config.unmapped_policy == "error":
        raise ValueError(f"unmapped genes found during preprocessing: {n_unmapped}")

    canonical_filled = canonical_ids.astype("object").copy()
    if config.unmapped_policy == "keep_original":
        canonical_filled.loc[unmapped_mask] = adata.var_names.astype(str)[unmapped_mask.to_numpy()]

    keep_mask = ~unmapped_mask if config.unmapped_policy == "drop" else pd.Series(True, index=canonical_filled.index)
    kept_ids = canonical_filled.loc[keep_mask].astype(str).reset_index(drop=True)
    kept_symbols = gene_symbols.loc[keep_mask].astype(str).reset_index(drop=True)
    kept_adata = adata[:, keep_mask.to_numpy()].copy()

    counts = kept_ids.value_counts()
    duplicate_ids = counts[counts > 1].index.tolist()
    n_duplicate = int(sum(counts[counts > 1] - 1))
    if duplicate_ids and config.duplicate_policy == "error":
        raise ValueError(f"duplicate canonical gene IDs found: {len(duplicate_ids)}")

    ordered_unique = pd.Index(kept_ids).drop_duplicates(keep="first")
    groups = [np.where(kept_ids.to_numpy() == gene_id)[0] for gene_id in ordered_unique]
    x_new = _aggregate_matrix(kept_adata.X, groups, config.duplicate_policy)
    layers_new = _aggregate_layers({name: kept_adata.layers[name] for name in kept_adata.layers.keys()}, groups, config.duplicate_policy)

    var_rows = []
    for gene_id, idx in zip(ordered_unique, groups):
        first = int(idx[0])
        row = kept_adata.var.iloc[first].copy()
        row[config.canonical_gene_id_column] = gene_id
        row[config.gene_symbol_column] = kept_symbols.iloc[first]
        row["atlasmtl_original_var_name"] = str(kept_adata.var_names[first])
        row["atlasmtl_duplicate_group_size"] = int(len(idx))
        var_rows.append(row)
    var_new = pd.DataFrame(var_rows, index=ordered_unique)
    var_new.index.name = None

    canonical_adata = AnnData(
        X=x_new,
        obs=kept_adata.obs.copy(),
        var=var_new,
        obsm={key: value.copy() for key, value in kept_adata.obsm.items()},
        uns=dict(kept_adata.uns),
        layers=layers_new,
    )
    canonical_adata.var_names = ordered_unique.astype(str)

    report = PreprocessReport(
        n_input_genes=n_input,
        n_canonical_genes=int(canonical_adata.n_vars),
        n_duplicate_genes=n_duplicate,
        n_unmapped_genes=n_unmapped,
        n_features_selected=int(canonical_adata.n_vars),
        feature_space=config.feature_space,
        species=config.species,
        var_names_type=config.var_names_type,
        mapping_resource=str(Path(config.gene_id_table or default_gene_id_table_path()).resolve()),
        duplicate_policy=config.duplicate_policy,
        unmapped_policy=config.unmapped_policy,
        ensembl_versions_stripped=int(stripped_count),
    )
    return canonical_adata, report
