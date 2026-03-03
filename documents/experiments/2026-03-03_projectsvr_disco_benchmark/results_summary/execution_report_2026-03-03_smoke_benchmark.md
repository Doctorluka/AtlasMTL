# DISCO first-wave smoke benchmark execution report

Date: `2026-03-03`

## Scope

This run used the prepared `5k` reference / `1k` heldout DISCO assets to test
whether the benchmark flow can execute across the full comparator set.

## Runtime assets

- prepared reference:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-03/reference_heldout/DISCO_hPBMCs/prepared/group_split_v1_train5k_test1k/reference_train.h5ad`
- prepared heldout:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-03/reference_heldout/DISCO_hPBMCs/prepared/group_split_v1_train5k_test1k/heldout_test.h5ad`
- smoke output root:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-03/reference_heldout/DISCO_hPBMCs/smoke_all_methods`
- run status:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-03/reference_heldout/DISCO_hPBMCs/smoke_all_methods/smoke_status.json`

## Comparator status

- all seven configured methods completed successfully:
  - `atlasmtl`
  - `reference_knn`
  - `celltypist`
  - `scanvi`
  - `singler`
  - `symphony`
  - `azimuth`

## Key smoke metrics (`cell_subtype`)

| method | accuracy | macro-F1 | coverage | covered-accuracy | notes |
|---|---:|---:|---:|---:|---|
| `atlasmtl` | `0.753` | `0.587` | `0.927` | `0.812` | moderate abstention (`unknown_rate=0.073`) |
| `reference_knn` | `0.542` | `0.453` | `1.000` | `0.542` | full coverage baseline |
| `celltypist` | `0.830` | `0.764` | `1.000` | `0.830` | ran on CellTypist-specific log1p inputs |
| `scanvi` | `0.798` | `0.607` | `1.000` | `0.798` | count-layer path worked |
| `singler` | `0.760` | `0.701` | `0.981` | `0.775` | mild reject behavior |
| `symphony` | `0.760` | `0.599` | `1.000` | `0.760` | completed via R backend |
| `azimuth` | `0.764` | `0.603` | `1.000` | `0.764` | backend recorded as `seurat_anchor_transfer_fallback` |

## Discussion

- The benchmark workflow is now operational end-to-end for DISCO, including the
  originally critical path where upstream counts lived in `adata.X` before
  preprocessing promoted them into `layers["counts"]`.
- DISCO is much easier than PH-Map on this smoke split, so most methods produce
  non-trivial subtype accuracy without obvious failure modes.
- This makes DISCO a good first-wave regression dataset for future benchmark
  code changes.
