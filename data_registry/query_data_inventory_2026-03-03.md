# Query data inventory (normalized)

Date: `2026-03-03`

This document is the normalized, repo-tracked version of the external source
note:

- `/home/data/fhz/project/phmap_package/data/real_test/query_data_information.md`

Use this file as the working source of truth for query-side benchmark planning,
manifest writing, and protocol checks.

## Scope and interpretation

- This file records:
  - query dataset paths
  - observed AnnData structure
  - intended evaluation use
  - matrix semantics relevant to atlasmtl
- Several query datasets carry internal labels, but the external note explicitly
  states that final validation may rely on marker-based review rather than
  strict use of those labels as benchmark truth.
- Do not promote a query dataset to a formal label-based benchmark target
  unless the project owner confirms that the corresponding query-side labels
  are acceptable as evaluation truth.

## 1) PBMC query

- **Path**: `/home/data/fhz/project/phmap_package/data/real_test/ProjectSVR/query_data/pbmc_query.h5ad`
- **Shape**: `20886 × 33694`
- **Observed structure**:
  - `obs`: `orig.ident`, `nCount_RNA`, `nFeature_RNA`, `donor`, `nUMI`, `nGene`, `percent_mito`, `cell_type`, `res_0.80`, `cell_subtype`, `DF.classifications`
  - `var`: `name`
  - `var_names`: Symbol-like
  - `layers`: none
  - `obsm`: `X_harmony.umap`, `X_pca.umap`
  - `obsp`: none
- **Intended use**:
  - external validation query for `DISCO_hPBMCs`
  - can be used operationally for single-level or multi-level transfer output
  - final validation is expected to rely on marker review, not necessarily on
    treating `cell_type` / `cell_subtype` as formal benchmark truth
- **Matrix semantics**:
  - `adata.X` has been verified as count-like integer sparse matrix
  - absence of `layers["counts"]` is acceptable for current compatibility testing

## 2) Vento query (decidua)

- **Path**: `/home/data/fhz/project/phmap_package/data/real_test/ProjectSVR/query_data/decidua_query.h5ad`
- **Shape**: `20521 × 21211`
- **Observed structure**:
  - `obs`: `orig.ident`, `nCount_RNA`, `nFeature_RNA`, `percent.mt`, `DF.classifications`, `seurat_clusters`, `RNA_snn_res.0.6`, `cell_type`
  - `var`: `name`
  - `var_names`: Symbol-like
  - `layers`: none
  - `obsm`: none
  - `obsp`: none
- **Intended use**:
  - external validation query for `Vento`
  - operationally usable for transfer output, but not yet a confirmed formal
    multi-level benchmark query
  - final validation is expected to rely on marker review
- **Matrix semantics**:
  - `adata.X` has been verified as count-like integer sparse matrix
  - absence of `layers["counts"]` is acceptable for current compatibility testing

## 3) PH-Map query

- **Path**: `/home/data/fhz/project/phmap_package/data/real_test/query_data/query_PH.h5ad`
- **Shape**: `50142 × 20324`
- **Observed structure**:
  - `obs`: `sample`, `group`, `batch`, QC/count summary columns, `passing_mt`, `passing_nUMIs`, `passing_ngenes`
  - `var`: `gene_ids`, `feature_types`, `genome`, `mt`, `ribo`, `hb`, count summary columns
  - `var_names`: Symbol-like
  - `layers`: `counts`
  - `obsm`: none
  - `obsp`: none
- **Intended use**:
  - external validation query for PH-Map atlas
  - may be used for single-level or multi-level transfer output
  - final validation is expected to rely on marker review rather than formal
    query labels inside `obs`
- **Matrix semantics**:
  - `var["gene_ids"]` stores Ensembl-like IDs while `var_names` are Symbol-like
  - `adata.X` is count-like integer sparse matrix
  - `layers["counts"]` is present and matches the current formal raw-count contract

## 4) HLCA query (`GSE302339`)

- **Path**: `/home/data/fhz/project/phmap_package/data/real_test/query_data/hlca_query_GSE302339.h5ad`
- **Shape**: `217320 × 34507`
- **Observed structure**:
  - `obs`: `GSE_id`, `Gold`, `Subject group`
  - `var`: `ENSEMBL`, `Symbol`
  - `var_names`: mixed Symbol/Ensembl namespace
  - `layers`: `counts`
  - `obsm`: `X_scANVI`, `X_umap`
  - `obsp`: `connectivities`, `distances`
- **Intended use**:
  - external validation query for HLCA core
  - may be used for single-level or multi-level transfer output
  - final validation is expected to rely on marker review unless `Gold` is
    explicitly approved as benchmark truth by the project owner
- **Matrix semantics**:
  - `adata.X` is consistent with non-negative log-normalized expression
  - `layers["counts"]` is present and stores count-like integer sparse values
  - `var["ENSEMBL"]` stores Ensembl identifiers; `var_names` are mixed and
    should not be treated as canonical gene IDs without preprocessing

## 5) Operational notes for atlasmtl

- Query datasets currently split into two matrix-contract groups:
  - `adata.X` count-like, no `layers["counts"]`: `pbmc_query`, `decidua_query`
  - `layers["counts"]` available: `query_PH`, `hlca_query_GSE302339`
- `query_PH` is symbol-like in `var_names` with Ensembl-like IDs in
  `var["gene_ids"]`.
- `hlca_query_GSE302339` is the most protocol-complex query because its
  `var_names` are mixed Symbol/Ensembl while canonical IDs live in `var["ENSEMBL"]`.

