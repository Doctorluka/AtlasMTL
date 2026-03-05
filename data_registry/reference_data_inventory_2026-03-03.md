# Reference data inventory (normalized)

Date: `2026-03-03`

This document is the normalized, repo-tracked version of the external source
note:

- `/home/data/fhz/project/phmap_package/data/real_test/reference_data_infomation.md`

Use this file as the **working source of truth** for benchmark planning,
manifest writing, and protocol checks. The external file remains a raw human
note and may contain stale fields or manual transcription errors.

## Scope and interpretation

- This file records:
  - dataset paths
  - observed AnnData structure
  - intended benchmark/training use
  - matrix semantics relevant to atlasmtl
- Do **not** infer additional dataset uses beyond the ones explicitly recorded
  here without confirming with the project owner.
- Dataset cleaning itself is handled outside this document. This file only
  records the currently verified usable state.

## 1) PH-Map atlas

- **Path**: `/home/data/fhz/project/phmap_package/data/real_test/phmap/phmap_clean.h5ad`
- **Shape**: `234390 × 21977`
- **Observed structure**:
  - `obs`: `sample`, `dataset`, `study`, `group`, `tissue`, `anno_lv1`, `anno_lv2`, `anno_lv3`, `anno_lv4`
  - `var`: `Symbol`, `ENSEMBL`, `Synonym`
  - `var_names`: Symbol-like
  - `layers`: `counts`
  - `obsm`: `X_scANVI`, `X_umap`
  - `obsp`: `connectivities`, `distances`
- **Intended use**:
  - multi-level training: `anno_lv1`, `anno_lv2`, `anno_lv3`, `anno_lv4`
  - single-level coarse benchmark: `anno_lv2`
  - single-level fine benchmark: `anno_lv4`
- **Matrix semantics**:
  - raw counts are available in `layers["counts"]`
  - `adata.X` should not be assumed to be the formal raw-count contract
- **Sample-like grouping field**:
  - `sample`

## 2) HLCA core

- **Path**: `/home/data/fhz/project/phmap_package/data/real_test/HLCA/hlca_clean.h5ad`
- **Shape**: `584944 × 27402`
- **Observed structure**:
  - `obs`: `ann_level_1`, `ann_level_2`, `ann_level_3`, `ann_level_4`, `ann_level_5`
  - `var`: `Symbol`, `ENSEMBL`
  - `var_names`: Ensembl-like
  - `layers`: `counts`
  - `obsm`: `X_scANVI`, `X_umap`
  - `obsp`: `connectivities`, `distances`
  - `raw`: present
- **Important correction**:
  - `ann_finest_level` is **not** part of the current cleaned AnnData contract
    and should not be referenced in manifests or training configs.
- **Intended use**:
  - multi-level training: `ann_level_1`, `ann_level_2`, `ann_level_3`, `ann_level_4`, `ann_level_5`
  - single-level coarse benchmark: `ann_level_3`
  - single-level fine benchmark: `ann_level_5`
- **Matrix semantics**:
  - `adata.X` is consistent with non-negative log-normalized expression
  - `layers["counts"]` exists and should be used as the formal counts contract
  - `adata.raw.X` is count-like and can be treated as a provenance check, not
    the primary runtime counts path unless explicitly needed
- **Sample-like grouping field**:
  - `donor_id`

## 3) mTCA

- **Path**: `/home/data/fhz/project/phmap_package/data/real_test/ProjectSVR/reference_atlas/mTCA.h5ad`
- **Shape**: `188862 × 32285`
- **Observed structure**:
  - `obs`: `orig.ident`, `Cell_type_level1`, `Cell_type_level2`, `Cell_type_level3`, `Cell_type_final`
  - `var`: `name`
  - `var_names`: Symbol-like
  - `obsm`: `X_umap`
  - `layers`: none
- **Species**:
  - mouse
- **Intended use**:
  - multi-level training: `Cell_type_level1`, `Cell_type_level2`, `Cell_type_level3`
  - single-level coarse benchmark: `Cell_type_level2`
  - single-level fine benchmark: `Cell_type_level3`
- **Matrix semantics**:
  - `adata.X` has been manually verified as count-like integer sparse matrix
  - this dataset is intentionally kept **without** `layers["counts"]` to test
    whether the current atlasmtl flow correctly handles the case where counts
    are stored in `adata.X`
- **Sample-like grouping field**:
  - `orig.ident`

## 4) DISCO PBMC

- **Path**: `/home/data/fhz/project/phmap_package/data/real_test/ProjectSVR/reference_atlas/DISCO_hPBMCs.h5ad`
- **Shape**: `167594 × 33538`
- **Observed structure**:
  - `obs`: `sample`, `cell_subtype`, `cell_type`
  - `var`: `name`
  - `var_names`: Symbol-like
  - `obsm`: `X_scvi`, `X_umap`
  - `layers`: none
- **Intended use**:
  - not used for hierarchical training
  - single-level coarse benchmark: `cell_type`
  - single-level fine benchmark: `cell_subtype`
- **Matrix semantics**:
  - `adata.X` was re-checked on `2026-03-04` and shows non-integer positive values
    consistent with a log-normalized matrix rather than raw counts
  - `layers["counts"]` is absent
  - this dataset is therefore currently **not ready** for formal atlasmtl-scale
    preprocessing until a valid raw-count source is provided
- **Sample-like grouping field**:
  - `sample`

## 5) CD4 T cell atlas

- **Path**: `/home/data/fhz/project/phmap_package/data/real_test/ProjectSVR/reference_atlas/cd4.h5ad`
- **Shape**: `150361 × 5095`
- **Observed structure**:
  - `obs`: `sample`, `cell_type`, `cell_subtype`
  - `var`: `name`
  - `var_names`: Symbol-like
  - `obsm`: `X_umap`
  - `layers`: none
- **Intended use**:
  - not used for hierarchical training
  - single-level coarse benchmark: `cell_type`
  - single-level fine benchmark: `cell_subtype`
- **Matrix semantics**:
  - re-checked on `2026-03-04`
  - `adata.X` remains consistent with count-like integer sparse values
  - `layers["counts"]` is absent by design
  - still suitable for validating auto-handling of counts stored in `adata.X`
- **Sample-like grouping field**:
  - `sample`

## 6) CD8 T cell atlas

- **Path**: `/home/data/fhz/project/phmap_package/data/real_test/ProjectSVR/reference_atlas/cd8.h5ad`
- **Shape**: `225212 × 4549`
- **Observed structure**:
  - `obs`: `sample`, `cell_type`, `cell_subtype`
  - `var`: `name`
  - `var_names`: Symbol-like
  - `obsm`: `X_umap`
  - `layers`: none
- **Intended use**:
  - not used for hierarchical training
  - single-level coarse benchmark: `cell_type`
  - single-level fine benchmark: `cell_subtype`
- **Matrix semantics**:
  - re-checked on `2026-03-04`
  - `adata.X` contains non-integer positive values in later rows, consistent
    with a normalized matrix rather than strict raw counts
  - `layers["counts"]` is absent and `raw` is absent
  - this dataset is therefore currently **not ready** for formal atlasmtl-scale
    preprocessing until a valid raw-count source is provided
- **Sample-like grouping field**:
  - `sample`

## 7) Vento

- **Path**: `/home/data/fhz/project/phmap_package/data/real_test/ProjectSVR/reference_atlas/vento.h5ad`
- **Shape**: `70325 × 31010`
- **Observed structure**:
  - `obs`: `orig.ident`, `annotation`
  - `var`: `name`
  - `var_names`: Symbol-like
  - `obsm`: `X_umap`
  - `layers`: none
- **Important correction**:
  - this dataset does **not** currently provide separate coarse/fine labels
  - do **not** treat it as a `cell_type` / `cell_subtype` two-level benchmark
- **Intended use**:
  - single-label reference dataset only, using `annotation`
- **Matrix semantics**:
  - re-checked on `2026-03-04`
  - `adata.X` remains consistent with count-like integer sparse values
  - `layers["counts"]` is absent by design
  - still suitable for validating auto-handling of counts stored in `adata.X`
- **Sample-like grouping field**:
  - `orig.ident`

## 8) Operational notes for atlasmtl

- For PH-Map and HLCA, the formal raw-count contract is `layers["counts"]`.
- For the four ProjectSVR-derived reference atlases (`mTCA`, `DISCO`, `cd4`,
  `cd8`) and `Vento`, `adata.X` is currently the verified counts store and the
  absence of `layers["counts"]` is intentional for compatibility testing.
- If a future cleaned version adds `layers["counts"]`, update this document and
  explicitly state whether those datasets should still be used for the
  `adata.X` auto-handling validation path.
