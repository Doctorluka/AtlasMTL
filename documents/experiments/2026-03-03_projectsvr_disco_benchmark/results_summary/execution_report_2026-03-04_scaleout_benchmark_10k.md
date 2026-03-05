# DISCO second-wave scale-out benchmark (`10k` heldout)

- scenario: `DISCO_hPBMCs` reference-heldout scale-out benchmark
- manifests:
  - `documents/experiments/2026-03-03_projectsvr_disco_benchmark/manifests/reference_heldout/DISCO_hPBMCs__cell_subtype__scaleout_runtime_10k_v1.yaml`
- output root: `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/DISCO_hPBMCs/benchmark/group_split_v2_train100k_test10k/all_methods_v2/`
- methods attempted: `atlasmtl`, `reference_knn`, `celltypist`, `scanvi`, `singler`, `symphony`, `seurat_anchor_transfer`
- run status: `7/7 success`

## Metric summary

- `atlasmtl`: `accuracy=0.8895`, `macro_f1=0.8129`, `coverage=0.9963`, `unknown_rate=0.0037`
- `scanvi`: `accuracy=0.8834`, `macro_f1=0.8231`
- `celltypist`: `accuracy=0.7830`, `macro_f1=0.7182`
- `singler`: `accuracy=0.7834`, `macro_f1=0.7431`, `unknown_rate=0.0184`
- `symphony`: `accuracy=0.7726`, `macro_f1=0.6582`
- `reference_knn`: `accuracy=0.7217`, `macro_f1=0.6143`
- `seurat_anchor_transfer`: `accuracy=0.7327`, `macro_f1=0.5490`, backend=`seurat_anchor_transfer_transferdata`

## Benchmark resource summary

| Method | Train time (s) | Predict time (s) | Train peak RSS (GB) | Predict peak RSS (GB) |
| --- | ---: | ---: | ---: | ---: |
| `atlasmtl` | `16.3469` | `0.1123` | `5.2509` | `5.3386` |
| `reference_knn` | `2.4425` | `8.4492` | `4.6697` | `4.6697` |
| `celltypist` | `0.0012` | `0.4033` | `1.0145` | `1.3378` |
| `scanvi` | `784.6213` | `29.6214` | `3.1783` | `3.1783` |
| `singler` | `1300.5495` | `1300.5495` | `0.0` | `0.0` |
| `symphony` | `70.5704` | `70.5704` | `0.0` | `0.0` |
| `seurat_anchor_transfer` | `6525.8052` | `6525.8052` | `0.0` | `0.0` |

Preparation-stage resource summary for the same dataset:

| Stage | Elapsed (s) | Avg RSS (GB) | Peak RSS (GB) |
| --- | ---: | ---: | ---: |
| `prepare_reference_heldout_scaleout` | `70.6514` | `19.2704` | `23.9861` |

## Discussion

- second-wave full benchmark flow is now verified on a `100k build / 10k heldout` real reference scenario
- `DISCO` remains the strongest engineering regression dataset for the current benchmark framework
- `atlasmtl` and `scanvi` are both strong on this setting; `atlasmtl` keeps a very low reject rate while preserving the best raw accuracy in this run
- `celltypist` required dedicated comparator preprocessing and a per-run trained model; this is now handled by the scale-out wrapper rather than the generic runner
- the current `celltypist` timing in this report still reflects the lightweight wrapped-logistic trainer, not the historical formal `celltypist.train(...)` workflow
- `seurat_anchor_transfer` is operational, but its stable backend remains `TransferData` rather than a `MapQuery` projection path
- external-comparator RSS accounting is still incomplete for `singler`, `symphony`, and `seurat_anchor_transfer`; the `0.0` values should be treated as monitoring gaps, not true zero-cost runs
