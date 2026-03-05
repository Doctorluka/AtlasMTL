# PH-Map second-wave scale-out benchmark (`10k` heldout)

- scenario: `PHMap_Lung_Full_v43_light` reference-heldout scale-out benchmark
- manifests:
  - `documents/experiments/2026-03-03_phmap_benchmark/manifests/reference_heldout/PHMap_Lung_Full_v43_light__anno_lv4__scaleout_runtime_10k_v1.yaml`
- output root: `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/PHMap_Lung_Full_v43_light/benchmark/group_split_v2_train100k_test10k/all_methods_v1/`
- methods attempted: `atlasmtl`, `reference_knn`, `celltypist`, `scanvi`, `singler`, `symphony`, `seurat_anchor_transfer`
- run status: `7/7 success`

## Metric summary

- `scanvi`: `accuracy=0.6440`, `macro_f1=0.6665`
- `atlasmtl`: `accuracy=0.6404`, `macro_f1=0.6887`, `coverage=0.8616`, `unknown_rate=0.1384`, `covered_accuracy=0.7433`
- `singler`: `accuracy=0.5598`, `macro_f1=0.5727`, `coverage=0.9669`, `unknown_rate=0.0331`
- `seurat_anchor_transfer`: `accuracy=0.5529`, `macro_f1=0.4560`, backend=`seurat_anchor_transfer_transferdata`
- `celltypist`: `accuracy=0.5448`, `macro_f1=0.5712`
- `reference_knn`: `accuracy=0.4826`, `macro_f1=0.4846`
- `symphony`: `accuracy=0.4591`, `macro_f1=0.4307`

## Benchmark resource summary

| Method | Train time (s) | Predict time (s) | Train peak RSS (GB) | Predict peak RSS (GB) |
| --- | ---: | ---: | ---: | ---: |
| `atlasmtl` | `16.2554` | `0.0840` | `5.3958` | `6.5420` |
| `reference_knn` | `2.4573` | `8.3690` | `4.7412` | `4.7412` |
| `celltypist` | `0.0014` | `0.4164` | `1.1803` | `1.3412` |
| `scanvi` | `1001.9616` | `36.7382` | `3.3929` | `3.3929` |
| `singler` | `1470.6664` | `1470.6664` | `0.0` | `0.0` |
| `symphony` | `67.7206` | `67.7206` | `0.0` | `0.0` |
| `seurat_anchor_transfer` | `2837.4952` | `2837.4952` | `0.0` | `0.0` |

Preparation-stage resource summary for the same dataset:

| Stage | Elapsed (s) | Avg RSS (GB) | Peak RSS (GB) |
| --- | ---: | ---: | ---: |
| `prepare_reference_heldout_scaleout` | `121.0114` | `32.7637` | `49.3992` |

## Discussion

- `PH-Map anno_lv4` remains a materially harder fine-label benchmark than `DISCO`
- `scanvi` and `atlasmtl` are the strongest methods on raw accuracy in this run, while `atlasmtl` keeps the strongest abstention behavior
- `atlasmtl` sacrifices coverage to keep a higher covered accuracy, which is consistent with the intended Unknown policy rather than a pipeline failure
- preparation cost is substantially larger than `DISCO`, especially in peak memory
- the current `celltypist` timing in this report still reflects the lightweight wrapped-logistic trainer, not the historical formal `celltypist.train(...)` workflow
- external-comparator RSS accounting is still incomplete for `singler`, `symphony`, and `seurat_anchor_transfer`; the `0.0` values should be treated as monitoring gaps
