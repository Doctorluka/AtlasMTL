# atlasmtl Preprocessing Design

## Scope

This document describes the dedicated preprocessing layer introduced for
atlasmtl. The preprocessing layer sits before `build_model()` and `predict()`
and is responsible for data-contract normalization rather than model training
or inference.

## Responsibilities

The preprocessing layer handles:

- gene namespace normalization
- counts-vs-non-counts input validation
- raw-count layer enforcement
- species-aware symbol / Ensembl mapping
- Ensembl version stripping
- duplicate and unmapped gene handling
- reference-defined feature panel construction
- HVG or whole-matrix feature-space selection
- preprocessing metadata traceability

## Canonical internal namespace

The canonical internal computation key is:

- versionless Ensembl gene ID

Readable symbols should be preserved in:

- `adata.var["gene_symbol"]`

## Public preprocessing API

- `preprocess_reference(adata, config)`
- `preprocess_query(adata, feature_panel, config)`

Supporting types:

- `PreprocessConfig`
- `FeaturePanel`
- `PreprocessReport`

## Default protocol

- `var_names_type`: explicit user input
- `species`: explicit user input
- if `adata.X` is count-like, copy it into `adata.layers["counts"]`
- if `adata.X` is not count-like, require `adata.layers["counts"]`
- atlasmtl core preprocessing does not run `normalize_total` or `log1p`
- default feature space: `hvg`
- default `n_top_genes`: `3000`
- default HVG method: `seurat_v3` with `layer="counts"`
- whole matrix requires explicit opt-in

## Metadata contract

The preprocessing layer writes metadata into:

- `adata.uns["atlasmtl_preprocess"]`
- `TrainedModel.train_config["preprocess"]`
- model manifests when available
- benchmark outputs and run manifests when preprocessing is configured

## Design boundary

`build_model()` and `predict()` do not silently guess the gene namespace and do
not silently recompute feature panels. They consume preprocessed data and
record the resulting metadata.
