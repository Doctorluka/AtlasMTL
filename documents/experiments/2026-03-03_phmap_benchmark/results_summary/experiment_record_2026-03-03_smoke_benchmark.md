# PH-Map first-wave smoke benchmark experiment record

Date: `2026-03-03`

## Design intent

Validate that the prepared PH-Map heldout assets can execute across the full
current comparator set and produce standardized benchmark outputs.

## Key settings

- dataset: `PHMap_Lung_Full_v43_light`
- label: `anno_lv4`
- prepared split: `5k` reference / `1k` heldout
- device: `cpu`
- methods:
  `atlasmtl`, `reference_knn`, `celltypist`, `scanvi`, `singler`, `symphony`, `azimuth`
- output root:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-03/reference_heldout/PHMap_Lung_Full_v43_light/smoke_all_methods`

## Implementation detail added during run

- A resilient smoke orchestrator was added:
  `documents/experiments/common/run_reference_heldout_smoke_benchmark.py`
- rationale:
  - the base benchmark runner aborts on first comparator failure
  - smoke execution needs method-by-method status capture

## Error and resolution

- initial CellTypist run failed because the prepared benchmark assets do not
  provide CellTypist-compatible log1p-normalized `X`
- fix:
  - generate comparator-specific reference/query copies from `layers["counts"]`
  - apply total-count normalization to `1e4`
  - apply `log1p`
  - train and run CellTypist on those dedicated assets only

## Notable observations

- all configured methods now finish successfully
- `azimuth` completed through `seurat_anchor_transfer_fallback`, not the native
  Azimuth backend
- `atlasmtl` shows strong abstention on this split, so covered accuracy is more
  informative than raw accuracy in this smoke round

## Next-round implications

- keep the CellTypist-specific log1p comparator-input path
- preserve per-method stdout/stderr logs under each `runs/<method>/`
- for formal reporting, add stronger heldout label-support constraints before
  interpreting PH-Map fine-label differences
