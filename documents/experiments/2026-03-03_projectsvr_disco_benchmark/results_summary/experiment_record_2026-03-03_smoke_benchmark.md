# DISCO first-wave smoke benchmark experiment record

Date: `2026-03-03`

## Design intent

Validate that the prepared DISCO heldout assets can execute across the full
current comparator set and produce standardized benchmark outputs.

## Key settings

- dataset: `DISCO_hPBMCs`
- label: `cell_subtype`
- prepared split: `5k` reference / `1k` heldout
- device: `cpu`
- methods:
  `atlasmtl`, `reference_knn`, `celltypist`, `scanvi`, `singler`, `symphony`, `seurat_anchor_transfer`
- output root:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-03/reference_heldout/DISCO_hPBMCs/smoke_all_methods_seurat_refactor`

## Important contract validated

- the DISCO source dataset originally relied on count-like `adata.X`
- preprocessing promoted those counts into `layers["counts"]`
- downstream comparators that require counts layers then ran without custom
  dataset-specific hacks

## Reused infrastructure

- smoke orchestration:
  `documents/experiments/common/run_reference_heldout_smoke_benchmark.py`
- CellTypist comparator-input preparation:
  comparator-specific log1p-normalized copies derived from `layers["counts"]`

## Error and resolution

- initial `seurat_anchor_transfer` rerun failed on DISCO because Seurat
  reference integration was unstable under the smoke-scale reference split and
  many-sample structure
- fixes applied:
  - cap PCA dimensions more aggressively for small per-batch objects
  - if Seurat reference integration fails, fall back to a single-reference PCA
    build
  - if `MapQuery`/UMAP-style projection is unavailable or unstable, fall back
    to `TransferData`

## Notable observations

- all configured methods completed successfully
- `celltypist`, `scanvi`, `singler`, `symphony`, and `seurat_anchor_transfer` all reached
  subtype accuracy around or above `0.76`
- `atlasmtl` showed useful abstention rather than obvious collapse
- the final DISCO Seurat comparator result used the
  `seurat_anchor_transfer_transferdata` backend after the above fallbacks

## Next-round implications

- DISCO should remain one of the default regression smoke datasets for the
  benchmark pipeline
- if future comparator wrappers change, rerun this dossier first before scaling
  to larger or harder atlases
