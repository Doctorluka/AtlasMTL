# HLCA formal third-wave benchmark execution report (`train10k`, `test5k`)

## Scenario

- dataset: `HLCA_Core`
- label: `ann_level_5`
- train: `10k`
- heldout predict: `5k`

## Group policy

- CPU group methods:
  - `atlasmtl`, `reference_knn`, `celltypist`, `singler`, `symphony`, `seurat_anchor_transfer`
- GPU group methods:
  - `atlasmtl`, `scanvi`

## Fairness policy

- CPU main-table policy: `cpu_only_strict`
- fixed thread env: `OMP=8`, `MKL=8`, `OPENBLAS=8`, `NUMEXPR=8`
- `scanvi` excluded from CPU group by round policy

## Runtime output paths

- prepared:
  - `/tmp/atlasmtl_benchmarks/2026-03-05/reference_heldout/HLCA_Core/prepared/formal_train10k_test5k/`
- CPU group:
  - `/tmp/atlasmtl_benchmarks/2026-03-05/reference_heldout/HLCA_Core/benchmark/formal_train10k_test5k/cpu_group_v1/`
- GPU group:
  - `/tmp/atlasmtl_benchmarks/2026-03-05/reference_heldout/HLCA_Core/benchmark/formal_train10k_test5k/gpu_group_v1/`

## Completion checklist

- [x] environment snapshots written
- [x] train10k materialization completed
- [x] CPU group completed
- [x] GPU group completed
- [x] summary tables generated from `summary.csv` and `metrics.json`

## Run status

- CPU group: `6/6` success
- GPU group: `2/2` success (rerun on CUDA-available shell)

## CPU metrics snapshot (`ann_level_5`)

| method | accuracy | macro_f1 | train_elapsed_s | predict_elapsed_s |
| --- | ---: | ---: | ---: | ---: |
| atlasmtl | 0.8492 | 0.5432 | 2.3156 | 0.0544 |
| reference_knn | 0.7766 | 0.4942 | 0.1331 | 1.2175 |
| celltypist (formal native) | 0.8626 | 0.7384 | 0.0018 | 0.2548 |
| singler | 0.8026 | 0.6863 | 92.7538 | 92.7538 |
| symphony | 0.6976 | 0.4192 | 18.8356 | 18.8356 |
| seurat_anchor_transfer | 0.8112 | 0.4563 | 84.8176 | 84.8176 |

## GPU metrics snapshot (`ann_level_5`)

| method | accuracy | macro_f1 | train_elapsed_s | predict_elapsed_s |
| --- | ---: | ---: | ---: | ---: |
| atlasmtl (cuda) | 0.8384 | 0.5274 | 2.0867 | 0.0446 |
| scanvi (cuda) | 0.8734 | 0.7016 | 54.0555 | 13.8746 |

## Resource-monitoring note

- Python-native methods reported RSS/throughput fields as expected.
- R comparator runs (`singler`, `symphony`, `seurat_anchor_transfer`) still
  show incomplete RSS/core-equivalent fields under current monitoring path.
- `joblib` serial fallback warning still appears in stderr in this environment;
  keep this run marked as fairness-degraded for strict runtime interpretation.
