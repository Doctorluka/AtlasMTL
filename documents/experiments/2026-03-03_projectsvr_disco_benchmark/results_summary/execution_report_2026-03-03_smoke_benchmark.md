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
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-03/reference_heldout/DISCO_hPBMCs/smoke_all_methods_seurat_refactor`
- run status:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-03/reference_heldout/DISCO_hPBMCs/smoke_all_methods_seurat_refactor/smoke_status.json`

## Comparator status

- all seven configured methods completed successfully:
  - `atlasmtl`
  - `reference_knn`
  - `celltypist`
  - `scanvi`
  - `singler`
  - `symphony`
  - `seurat_anchor_transfer`

## Key smoke metrics (`cell_subtype`)

| method | accuracy | macro-F1 | coverage | covered-accuracy | notes |
|---|---:|---:|---:|---:|---|
| `atlasmtl` | `0.761` | `0.622` | `0.908` | `0.838` | moderate abstention (`unknown_rate=0.092`) |
| `reference_knn` | `0.542` | `0.453` | `1.000` | `0.542` | full coverage baseline |
| `celltypist` | `0.829` | `0.764` | `1.000` | `0.829` | ran on CellTypist-specific log1p inputs |
| `scanvi` | `0.783` | `0.575` | `1.000` | `0.783` | count-layer path worked |
| `singler` | `0.760` | `0.701` | `0.981` | `0.775` | mild reject behavior |
| `symphony` | `0.760` | `0.599` | `1.000` | `0.760` | completed via R backend |
| `seurat_anchor_transfer` | `0.751` | `0.558` | `1.000` | `0.751` | backend = `seurat_anchor_transfer_transferdata` after integration fallback |

## Discussion

- The benchmark workflow is now operational end-to-end for DISCO, including the
  originally critical path where upstream counts lived in `adata.X` before
  preprocessing promoted them into `layers["counts"]`.
- The first Seurat refactor pass exposed that DISCO's many-sample reference can
  make Seurat reference integration unstable in smoke-scale settings. The
  comparator now falls back to a single-reference PCA build and then to
  `TransferData` when `MapQuery`-style projection is not stable.
- DISCO is much easier than PH-Map on this smoke split, so most methods produce
  non-trivial subtype accuracy without obvious failure modes.
- This makes DISCO a good first-wave regression dataset for future benchmark
  code changes.
