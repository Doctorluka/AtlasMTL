# mTCA second-wave scale-out benchmark (`10k` heldout)

- scenario: `mTCA` reference-heldout scale-out benchmark
- manifests:
  - `documents/experiments/2026-03-04_projectsvr_mtca_benchmark/manifests/reference_heldout/mTCA__Cell_type_level3__scaleout_runtime_10k_v1.yaml`
- output root: `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/mTCA/benchmark/group_split_v2_train100k_test10k/all_methods_v2_mouse_fix/`
- methods attempted: `atlasmtl`, `reference_knn`, `celltypist`, `scanvi`, `singler`, `symphony`, `seurat_anchor_transfer`
- run status: `7/7 success`

## Metric summary

- `atlasmtl`: `accuracy=0.9442`, `macro_f1=0.8814`, `coverage=0.9960`, `unknown_rate=0.0040`
- `scanvi`: `accuracy=0.9437`, `macro_f1=0.8666`
- `celltypist`: `accuracy=0.9081`, `macro_f1=0.8457`
- `seurat_anchor_transfer`: `accuracy=0.8933`, `macro_f1=0.7723`, backend=`seurat_anchor_transfer_transferdata`
- `singler`: `accuracy=0.8513`, `macro_f1=0.7861`, `unknown_rate=0.0201`
- `reference_knn`: `accuracy=0.8045`, `macro_f1=0.6602`
- `symphony`: `accuracy=0.6417`, `macro_f1=0.5356`

## Benchmark resource summary

| Method | Train time (s) | Predict time (s) | Train peak RSS (GB) | Predict peak RSS (GB) |
| --- | ---: | ---: | ---: | ---: |
| `atlasmtl` | `16.7490` | `0.1797` | `5.2143` | `5.3683` |
| `reference_knn` | `2.8380` | `5.0840` | `4.7300` | `4.7300` |
| `celltypist` | `0.0012` | `0.4770` | `1.3278` | `1.4256` |
| `scanvi` | `856.9848` | `43.0450` | `3.3888` | `3.3888` |
| `singler` | `1645.0021` | `1645.0021` | `0.0` | `0.0` |
| `symphony` | `62.5601` | `62.5601` | `0.0` | `0.0` |
| `seurat_anchor_transfer` | `190.5305` | `190.5305` | `0.0` | `0.0` |

Preparation-stage resource summary for the same dataset:

| Stage | Elapsed (s) | Avg RSS (GB) | Peak RSS (GB) |
| --- | ---: | ---: | ---: |
| `prepare_reference_heldout_scaleout` | `91.6677` | `37.8894` | `52.5326` |

## Discussion

- the first `mTCA` benchmark attempt was invalid because the prep manifest used `species=human` for a mouse dataset; that run should not be interpreted
- after correcting the manifest to `species=mouse`, the feature space recovered from an invalid `19` genes to a valid `3000 HVG` reference panel and all comparators became runnable
- `atlasmtl` and `scanvi` now form the top tier on this scenario and are nearly tied on raw accuracy
- `celltypist` remains strong on this dataset, but the current benchmark result still reflects the lightweight `wrapped_logreg` trainer path rather than the historical formal `celltypist.train(...)` workflow
- `singler` and `symphony` both recovered from the previous failed run once the species/gene-space issue was fixed, which confirms the earlier failures were contract problems rather than standalone comparator breakage
- external-comparator RSS accounting remains incomplete for `singler`, `symphony`, and `seurat_anchor_transfer`; the `0.0` values should be treated as monitoring gaps, not real zero-cost usage
