# PH-Map first-wave smoke benchmark execution report

Date: `2026-03-03`

## Scope

This run used the prepared `5k` reference / `1k` heldout PH-Map assets to test
whether the benchmark flow can execute across the full comparator set.

## Runtime assets

- prepared reference:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-03/reference_heldout/PHMap_Lung_Full_v43_light/prepared/group_split_v1_train5k_test1k/reference_train.h5ad`
- prepared heldout:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-03/reference_heldout/PHMap_Lung_Full_v43_light/prepared/group_split_v1_train5k_test1k/heldout_test.h5ad`
- smoke output root:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-03/reference_heldout/PHMap_Lung_Full_v43_light/smoke_all_methods_seurat_refactor`
- run status:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-03/reference_heldout/PHMap_Lung_Full_v43_light/smoke_all_methods_seurat_refactor/smoke_status.json`

## Comparator status

- all seven configured methods completed successfully:
  - `atlasmtl`
  - `reference_knn`
  - `celltypist`
  - `scanvi`
  - `singler`
  - `symphony`
  - `seurat_anchor_transfer`

## Key smoke metrics (`anno_lv4`)

| method | accuracy | macro-F1 | coverage | covered-accuracy | notes |
|---|---:|---:|---:|---:|---|
| `atlasmtl` | `0.414` | `0.334` | `0.536` | `0.772` | high abstention (`unknown_rate=0.464`) |
| `reference_knn` | `0.368` | `0.342` | `1.000` | `0.368` | full coverage baseline |
| `celltypist` | `0.613` | `0.576` | `1.000` | `0.613` | required CellTypist-specific log1p input preparation |
| `scanvi` | `0.538` | `0.392` | `1.000` | `0.538` | count-layer path worked |
| `singler` | `0.573` | `0.549` | `0.965` | `0.594` | mild reject behavior |
| `symphony` | `0.467` | `0.426` | `1.000` | `0.467` | completed via R backend |
| `seurat_anchor_transfer` | `0.450` | `0.364` | `1.000` | `0.450` | backend = `seurat_anchor_transfer_transferdata` |

## Discussion

- The benchmark workflow is now operational end-to-end for PH-Map under the
  renamed `seurat_anchor_transfer` comparator.
- The original CellTypist path failed because it expects log1p-normalized
  expression in `X`; this run fixed that by generating CellTypist-specific
  log1p-normalized comparator inputs from `layers["counts"]`.
- The Seurat comparator completed through the explicit `TransferData` backend
  rather than a `MapQuery` UMAP projection path. For this first-wave smoke run,
  that is acceptable because the goal is stable label transfer, not embedding
  projection fidelity.
- `atlasmtl` behaved conservatively on this split, with low raw coverage but
  much higher covered accuracy. For smoke validation this is acceptable and not
  obviously pathological.
- PH-Map remains a hard fine-label task on this sampled split. These numbers are
  suitable for process validation, not for paper-grade method ranking.
