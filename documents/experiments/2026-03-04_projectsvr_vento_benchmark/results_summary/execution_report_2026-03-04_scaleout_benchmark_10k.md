# Vento second-wave scale-out benchmark (`10k` heldout)

- scenario: `Vento` reference-heldout scale-out benchmark
- manifests:
  - `documents/experiments/2026-03-04_projectsvr_vento_benchmark/manifests/reference_heldout/Vento__annotation__scaleout_runtime_10k_v1.yaml`
- output root: `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/Vento/benchmark/group_split_v2_train50k_test10k/all_methods_v1/`
- methods attempted: `atlasmtl`, `reference_knn`, `celltypist`, `scanvi`, `singler`, `symphony`, `seurat_anchor_transfer`
- run status: `7/7 success`

## Metric summary

- `scanvi`: `accuracy=0.9110`, `macro_f1=0.8546`
- `atlasmtl`: `accuracy=0.9027`, `macro_f1=0.8586`, `coverage=0.9906`, `unknown_rate=0.0094`
- `celltypist`: `accuracy=0.8595`, `macro_f1=0.8030`
- `singler`: `accuracy=0.8525`, `macro_f1=0.8213`, `unknown_rate=0.0217`
- `reference_knn`: `accuracy=0.7393`, `macro_f1=0.5773`
- `symphony`: `accuracy=0.7230`, `macro_f1=0.5660`
- `seurat_anchor_transfer`: `accuracy=0.5212`, `macro_f1=0.3795`, backend=`seurat_anchor_transfer_transferdata`

## Benchmark resource summary

| Method | Train time (s) | Predict time (s) | Train peak RSS (GB) | Predict peak RSS (GB) |
| --- | ---: | ---: | ---: | ---: |
| `atlasmtl` | `8.0314` | `0.1219` | `3.2407` | `3.3040` |
| `reference_knn` | `1.2054` | `5.5247` | `2.8200` | `2.8200` |
| `celltypist` | `0.0012` | `0.3962` | `1.0177` | `1.3380` |
| `scanvi` | `563.6274` | `40.6870` | `2.3198` | `2.3198` |
| `singler` | `516.8435` | `516.8435` | `0.0` | `0.0` |
| `symphony` | `44.4714` | `44.4714` | `0.0` | `0.0` |
| `seurat_anchor_transfer` | `1182.1461` | `1182.1461` | `0.0` | `0.0` |

Additional comparator-training note:

- `celltypist` also required wrapper-side comparator model training before the
  benchmark run itself: `160.8984s` on the `50k` prepared reference

Preparation-stage resource summary for the same dataset:

| Stage | Elapsed (s) | Avg RSS (GB) | Peak RSS (GB) |
| --- | ---: | ---: | ---: |
| `prepare_reference_heldout_scaleout` | `54.2178` | `13.6095` | `16.5931` |

## Discussion

- `Vento` validates the reduced-ceiling second-wave path: `50k` build +
  `10k` heldout + nested `5k`
- `scanvi` achieved the best raw accuracy, while `atlasmtl` achieved the best
  macro-F1 with low reject rate and near-identical covered accuracy
- `celltypist` remained strong on this single-label setting, but its wrapper
  still incurs a separate comparator model-training stage that should be kept
  distinct from the benchmark-side `load_model` timing
- `singler` and `symphony` both completed successfully on `Vento`, which means
  the ProjectSVR count-in-`adata.X` compatibility path is working end-to-end
- `seurat_anchor_transfer` completed successfully but remained much weaker than
  the leading methods on this scenario
- external-comparator RSS accounting remains incomplete for `singler`,
  `symphony`, and `seurat_anchor_transfer`; the `0.0` values should be treated
  as monitoring gaps rather than true zero-memory runs
