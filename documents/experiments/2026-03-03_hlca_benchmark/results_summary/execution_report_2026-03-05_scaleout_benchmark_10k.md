# HLCA second-wave scale-out benchmark (`10k` heldout)

- scenario: `HLCA_Core` reference-heldout scale-out benchmark
- manifests:
  - `documents/experiments/2026-03-03_hlca_benchmark/manifests/reference_heldout/HLCA_Core__ann_level_5__scaleout_runtime_10k_v1.yaml`
- output root: `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/benchmark/group_split_v2_train100k_test10k/all_methods_v1/`
- methods attempted: `atlasmtl`, `reference_knn`, `celltypist`, `scanvi`, `singler`, `symphony`, `seurat_anchor_transfer`
- run status: `7/7 success` (recovered after interruption)

## Metric summary

- `scanvi`: `accuracy=0.8935`, `macro_f1=0.7742`
- `atlasmtl`: `accuracy=0.8863`, `macro_f1=0.7585`, `coverage=0.9925`, `unknown_rate=0.0075`
- `seurat_anchor_transfer`: `accuracy=0.8333`, `macro_f1=0.5132`, backend=`seurat_anchor_transfer_transferdata`
- `singler`: `accuracy=0.8177`, `macro_f1=0.7444`, `unknown_rate=0.0234`
- `reference_knn`: `accuracy=0.8137`, `macro_f1=0.5617`
- `celltypist`: `accuracy=0.7834`, `macro_f1=0.7382`
- `symphony`: `accuracy=0.7168`, `macro_f1=0.4612`

## Benchmark resource summary

| Method | Train time (s) | Predict time (s) | Train peak RSS (GB) | Predict peak RSS (GB) |
| --- | ---: | ---: | ---: | ---: |
| `atlasmtl` | `20.5366` | `0.1618` | `5.1346` | `5.4656` |
| `reference_knn` | `1.6319` | `10.5160` | `3.7686` | `3.7686` |
| `celltypist` | `0.0016` | `0.4560` | `1.0126` | `1.3332` |
| `scanvi` | `1135.4385` | `36.9241` | `3.1578` | `3.1578` |
| `singler` | `1601.9736` | `1601.9736` | `0.0` | `0.0` |
| `symphony` | `70.6167` | `70.6167` | `0.0` | `0.0` |
| `seurat_anchor_transfer` | `5254.7418` | `5254.7418` | `0.0` | `0.0` |

Preparation-stage resource summary for the same dataset:

| Stage | Elapsed (s) | Avg RSS (GB) | Peak RSS (GB) |
| --- | ---: | ---: | ---: |
| `prepare_reference_heldout_scaleout` | `382.3883` | `64.0343` | `86.7406` |

## Recovery note

- the original wrapper run was interrupted before writing top-level
  `scaleout_status.json`
- `seurat_anchor_transfer` was rerun separately and method outputs were
  completed under:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/benchmark/group_split_v2_train100k_test10k/all_methods_v1/runs/seurat_anchor_transfer/`
- `scaleout_status.json` was then reconstructed from per-method outputs to
  restore a complete benchmark status manifest

## Discussion

- HLCA is now a complete second-wave `100k/10k` benchmark dataset with all seven
  comparators present
- `scanvi` and `atlasmtl` are the strongest methods on raw accuracy in this run
- `seurat_anchor_transfer` is operational but very expensive on wall-clock time
  under this CPU setup
- external comparator RSS accounting remains incomplete for R-based methods; the
  `0.0` entries are monitoring gaps, not true zero-cost usage
