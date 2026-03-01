# 2026-03-01 counts-only preprocess contract

## Scope

This update narrows atlasmtl preprocessing to a counts-first contract and removes
the previous assumption that atlasmtl core preprocessing should materialize a
default log-normalized working matrix.

## Code changes

- added `atlasmtl/preprocess/matrix_semantics.py` for count-like matrix
  detection
- extended `PreprocessConfig`, `FeaturePanel`, and `PreprocessReport` with
  counts-contract metadata
- updated `preprocess_reference(...)` and `preprocess_query(...)` to:
  - validate whether `adata.X` is count-like
  - create `adata.layers["counts"]` from `adata.X` only when valid
  - fail when `adata.X` is not count-like and no valid counts layer is present
- updated HVG selection so `hvg_method="seurat_v3"` explicitly uses
  `layer="counts"`
- updated the `scanvi` benchmark method to require and use
  `layer="counts"`

## Metadata changes

`atlasmtl_preprocess` metadata now records:

- declared and detected input matrix type
- counts-layer availability and validation status
- counts layer name used by preprocessing
- HVG layer used for feature selection

`feature_panel` artifacts now record:

- `counts_layer`
- `hvg_layer_used`

## Testing changes

- added unit coverage for count-like detection and counts-layer enforcement
- extended integration coverage for:
  - preprocessing roundtrip with automatic counts-layer creation
  - HVG selection on `layer="counts"`
  - benchmark `scanvi` metadata showing `counts` usage

## Documentation sync

Updated:

- `README.md`
- `benchmark/README.md`
- `documents/design/preprocessing.md`
- `documents/protocols/experiment_protocol.md`
- `AGENTS.md`

These documents now describe atlasmtl preprocessing as a counts-only core
contract, with log-normalization treated as a comparator-specific or optional
downstream concern rather than a default atlasmtl preprocessing step.

## Benchmark manifest follow-up

The benchmark runner now also accepts explicit preprocessing declarations in
dataset manifests:

- `input_matrix_type`
- `counts_layer`

These fields are propagated into `PreprocessConfig`, benchmark metrics, and run
manifests so formal runs no longer rely only on implicit defaults.
