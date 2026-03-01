# atlasmtl Preprocessing Layer Execution Plan

Date: 2026-03-01
Project root: `/home/data/fhz/project/phmap_package/atlasmtl`

## Summary

This plan defines the execution path for adding a dedicated preprocessing layer
 to atlasmtl. The preprocessing layer is responsible for:

- canonical gene-ID normalization
- species-aware symbol/Ensembl conversion
- duplicate and unmapped gene handling
- reference-defined feature panel construction
- HVG or whole-matrix feature-space selection
- consistent reference/query preprocessing metadata

The main design rule is that preprocessing should be explicit and modular. It
should not be hidden inside `build_model()` or `predict()`.

## Objectives

1. Keep `build_model()` and `predict()` focused on model training and
   inference.
2. Standardize the internal gene namespace to versionless Ensembl IDs.
3. Make formal experiments default to reference-derived HVG features.
4. Preserve traceability by storing preprocessing metadata in:
   - `AnnData.uns`
   - `TrainedModel.train_config`
   - model manifests
   - benchmark outputs

## Current State

Current atlasmtl behavior:

- `extract_matrix()` aligns genes by exact `adata.var_names`
- `input_transform="binary"` remains the default phmap-style input transform
- no runtime symbol/Ensembl resolver exists
- no species-aware preprocessing contract exists
- no formal HVG feature-selection layer exists

Existing historical preprocessing logic in older code:

- feature subset by `var_genes`
- binary conversion
- label encoding
- canonical training-gene ordering

These ideas are still useful, but the implementation is not sufficient for the
current atlasmtl protocol.

## Decisions Locked For First Implementation

### Gene namespace

- internal canonical namespace: versionless Ensembl IDs
- readable symbols preserved in `adata.var["gene_symbol"]`
- bundled mapping baseline:
  - `atlasmtl/resources/gene_id/biomart_human_mouse_rat.tsv.gz`

### Supported species

- `human`
- `mouse`
- `rat`

### Feature-space policy

- default: `feature_space="hvg"`
- default: `n_top_genes=3000`
- default HVG method: `seurat_v3`
- whole matrix must be explicitly requested:
  - `feature_space="whole"`

### HVG batch handling

- `hvg_batch_key` is optional
- if provided, forward it to the HVG selection implementation
- if absent, compute HVGs on the full reference matrix

## Git Baseline Requirement

Before code reclassification or preprocessing implementation proceeds, create a
clean git checkpoint for the current benchmark/doc state. The preprocessing
layer should be developed on top of that baseline rather than mixed into the
existing untracked benchmark evolution.

Recommended order:

1. classify current dirty tree
2. commit benchmark/doc/protocol/resource changes
3. start preprocessing implementation from that checkpoint

## Target Module Layout

Add a new package:

- `atlasmtl/preprocess/`

Planned files:

- `atlasmtl/preprocess/__init__.py`
- `atlasmtl/preprocess/gene_ids.py`
- `atlasmtl/preprocess/features.py`
- `atlasmtl/preprocess/metadata.py`
- `atlasmtl/preprocess/pipeline.py`
- `atlasmtl/preprocess/types.py`

### Responsibilities

#### `gene_ids.py`

- detect and normalize input gene namespace
- strip Ensembl version suffixes
- load bundled or external mapping table
- map symbols to canonical Ensembl IDs
- preserve readable gene symbols
- implement duplicate and unmapped policies

#### `features.py`

- compute reference-defined HVG panel
- support explicit whole-matrix mode
- create feature panel metadata
- align query data to the reference feature panel

#### `metadata.py`

- build preprocessing metadata payloads
- standardize metadata written to `AnnData.uns`
- standardize metadata written to model artifacts

#### `types.py`

- `PreprocessConfig`
- `FeaturePanel`
- `PreprocessReport`

#### `pipeline.py`

- `preprocess_reference(...)`
- `preprocess_query(...)`

## Public Interfaces

### `PreprocessConfig`

Planned fields:

- `var_names_type`: `symbol | ensembl`
- `species`: `human | mouse | rat`
- `gene_id_table`: optional external mapping table path
- `strip_ensembl_version`: default `True`
- `feature_space`: `hvg | whole`
- `n_top_genes`: default `3000`
- `hvg_method`: default `seurat_v3`
- `hvg_batch_key`: optional
- `duplicate_policy`: `sum | mean | first | error`
- `unmapped_policy`: `drop | keep_original | error`
- `gene_symbol_column`: default `gene_symbol`
- `canonical_gene_id_column`: default `ensembl_gene_id`

### `preprocess_reference(...)`

Consumes raw reference `AnnData`, returns:

- preprocessed reference `AnnData`
- `FeaturePanel`
- `PreprocessReport`

### `preprocess_query(...)`

Consumes raw query `AnnData` plus `FeaturePanel`, returns:

- preprocessed query `AnnData`
- `PreprocessReport`

## Integration Plan

### `build_model()`

- consume already-preprocessed reference data
- record preprocessing metadata into `train_config`
- do not silently remap gene IDs or recompute feature panels

### `predict()`

- consume query data already normalized to the same preprocessing contract
- keep existing exact feature alignment logic
- add metadata checks and traceability, but not hidden remapping

### CLI

Add preprocessing options to training and prediction CLIs:

- `--var-names-type`
- `--species`
- `--gene-id-table`
- `--feature-space`
- `--n-top-genes`
- `--hvg-method`
- `--hvg-batch-key`
- `--duplicate-policy`
- `--unmapped-policy`

Training CLI should be allowed to preprocess reference data directly.
Prediction CLI should preprocess query data against the stored feature panel,
not compute a new panel.

### Benchmark runner

Extend dataset-manifest support for:

- `var_names_type`
- `species`
- `gene_id_table`
- `feature_space`
- `hvg_config`
- `duplicate_policy`
- `unmapped_policy`

Runner should call preprocessing explicitly before model training or inference.

## Edge Cases

The first implementation must handle:

1. Ensembl IDs with version suffixes
2. symbols that do not map
3. duplicate canonical Ensembl IDs
4. query datasets missing part of the reference feature panel
5. `hvg_batch_key` not found in `obs`
6. explicit whole-matrix runs that are large but still valid

## Tests

### Unit tests

- gene-ID mapping
- Ensembl version stripping
- duplicate handling
- unmapped handling
- HVG panel creation
- query alignment to feature panel

### Integration tests

- reference preprocessing → `build_model()`
- query preprocessing → `predict()`
- CLI smoke tests with preprocessing options
- benchmark manifest integration for preprocessing metadata

## Acceptance Criteria

The preprocessing layer is complete when:

1. atlasmtl has a dedicated preprocessing package
2. gene-ID normalization is explicit and traceable
3. versionless Ensembl is the default internal namespace
4. reference-derived 3000 HVG is the default formal feature space
5. whole matrix requires explicit opt-in
6. preprocessing metadata is stored in `AnnData`, model artifacts, and benchmark outputs
7. existing model training/prediction APIs remain stable
