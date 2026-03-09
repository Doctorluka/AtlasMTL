# HLCA formal third-wave benchmark execution report (`train10k`, `test5k`)

## Scenario

- dataset: `HLCA_Core`
- label: `ann_level_5`
- train: `10k`
- heldout predict: `5k`

## Historical status note

- this report records the historical `2026-03-05` HLCA formal pilot exactly as
  executed.
- it predates the final pre-formal parameter locks completed on `2026-03-06`
  (`scanvi`) and `2026-03-07` (`atlasmtl`).
- do not reuse the pilot's `atlasmtl` or `scanvi` training defaults for later
  formal runs; use the locked defaults in the current protocol and the `*_v2`
  manifests instead.

## Group policy

- CPU group methods:
  - `atlasmtl`, `reference_knn`, `celltypist`, `singler`, `symphony`, `seurat_anchor_transfer`
- GPU group methods:
  - `atlasmtl`, `scanvi`

## Fairness policy

- CPU main-table policy: `cpu_only_strict`
- fixed thread env: `OMP=8`, `MKL=8`, `OPENBLAS=8`, `NUMEXPR=8`
- `scanvi` excluded from CPU group by round policy
- machine-readable fairness fields are written into per-method `metrics.json`
  (`fairness_metadata`) and wrapper `scaleout_status.json`

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
| atlasmtl | 0.8376 | 0.5643 | 2.1903 | 0.0540 |
| reference_knn | 0.7766 | 0.4942 | 0.1328 | 1.2315 |
| celltypist (formal native) | 0.8618 | 0.7386 | 0.0018 | 0.2488 |
| singler | 0.8026 | 0.6863 | 92.8093 | 92.8093 |
| symphony | 0.6976 | 0.4192 | 18.8621 | 18.8621 |
| seurat_anchor_transfer | 0.8112 | 0.4563 | 85.4420 | 85.4420 |

## GPU metrics snapshot (`ann_level_5`)

| method | accuracy | macro_f1 | train_elapsed_s | predict_elapsed_s |
| --- | ---: | ---: | ---: | ---: |
| atlasmtl (cuda) | 0.8536 | 0.5658 | 1.4069 | 0.0288 |
| scanvi (cuda) | 0.8672 | 0.6422 | 36.5337 | 7.0884 |

`scanvi` runtime defaults used in this historical pilot:

- `scvi_max_epochs=15`
- `scanvi_max_epochs=15`
- `query_max_epochs=10`
- `datasplitter_num_workers=0`

Current locked formal default for later reruns:

- `scvi_max_epochs=25`
- `scanvi_max_epochs=25`
- `query_max_epochs=20`
- `n_latent=20`
- `batch_size=256`
- `datasplitter_num_workers=0`

## Resource-monitoring note

- Python-native methods report RSS/throughput and fairness metadata as expected.
- R comparator runs now include peak RSS via subprocess `/usr/bin/time` fallback.
- `joblib` serial fallback warning appears in CPU-group stderr, so CPU runtime is
  marked `runtime_fairness_degraded=true` in machine-readable metadata.
