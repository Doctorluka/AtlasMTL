# HLCA formal third-wave benchmark record (`train10k`, `test5k`)

## Run metadata

- date: `2026-03-05`
- operator: `codex`
- environment: `atlasmtl-env`

## Manifest references

- historical CPU manifest used in the `2026-03-05` run:
  - `documents/experiments/2026-03-05_formal_third_wave_hlca_benchmark/manifests/reference_heldout/HLCA_Core__ann_level_5__formal_runtime_cpu_train10k_test5k_v1.yaml`
- historical GPU manifest used in the `2026-03-05` run:
  - `documents/experiments/2026-03-05_formal_third_wave_hlca_benchmark/manifests/reference_heldout/HLCA_Core__ann_level_5__formal_runtime_gpu_train10k_test5k_v1.yaml`
- rerun-ready CPU manifest with locked formal defaults:
  - `documents/experiments/2026-03-05_formal_third_wave_hlca_benchmark/manifests/reference_heldout/HLCA_Core__ann_level_5__formal_runtime_cpu_train10k_test5k_v2.yaml`
- rerun-ready GPU manifest with locked formal defaults:
  - `documents/experiments/2026-03-05_formal_third_wave_hlca_benchmark/manifests/reference_heldout/HLCA_Core__ann_level_5__formal_runtime_gpu_train10k_test5k_v2.yaml`

## Commands

- split materialization:
  - `python documents/experiments/2026-03-05_formal_third_wave_hlca_benchmark/scripts/prepare_formal_hlca_train10k_test5k.py`
- environment snapshot:
  - `bash documents/experiments/2026-03-05_formal_third_wave_hlca_benchmark/scripts/record_environment_versions.sh`
- CPU group:
  - `bash documents/experiments/2026-03-05_formal_third_wave_hlca_benchmark/scripts/run_formal_hlca_cpu_group.sh`
- GPU group:
  - `bash documents/experiments/2026-03-05_formal_third_wave_hlca_benchmark/scripts/run_formal_hlca_gpu_group.sh`
- table export:
  - `python documents/experiments/2026-03-05_formal_third_wave_hlca_benchmark/scripts/export_formal_tables.py`

## Observations

- [x] all CPU group methods succeeded
- [x] all GPU group methods succeeded (after rerun on CUDA-available shell)
- [x] no missing CPU run artifacts (`summary.csv`, `metrics.json`, `scaleout_status.json`)
- [x] fallback paths are explicitly visible in method metadata (for example
  `seurat_anchor_transfer_transferdata`)

## Runtime artifacts

- prepared split root:
  - `/tmp/atlasmtl_benchmarks/2026-03-05/reference_heldout/HLCA_Core/prepared/formal_train10k_test5k/`
- CPU group root:
  - `/tmp/atlasmtl_benchmarks/2026-03-05/reference_heldout/HLCA_Core/benchmark/formal_train10k_test5k/cpu_group_v1/`
- GPU group root:
  - `/tmp/atlasmtl_benchmarks/2026-03-05/reference_heldout/HLCA_Core/benchmark/formal_train10k_test5k/gpu_group_v1/`

## Key outcomes

- `celltypist` train path confirmed as native formal backend on this run.
- CPU accuracy lead in this setting is from `celltypist` (`0.8626`), followed
  by `atlasmtl` (`0.8492`) on `ann_level_5`.
- initial GPU attempt failed under restricted execution, then succeeded after
  rerun in CUDA-available shell.

## Detailed record: scanvi epoch-lock experiment

### 1) Data used for this parameter test

- dataset: `HLCA_Core`
- train file:
  - `/tmp/atlasmtl_benchmarks/2026-03-05/reference_heldout/HLCA_Core/prepared/formal_train10k_test5k/reference_train_10k.h5ad`
- predict file:
  - `/tmp/atlasmtl_benchmarks/2026-03-05/reference_heldout/HLCA_Core/prepared/formal_train10k_test5k/heldout_test_5k.h5ad`
- label column: `ann_level_5`
- matrix contract: `counts_layer=counts`
- scale:
  - reference cells: `10,000`
  - heldout cells: `5,000`
  - genes/features: `3,000`
  - reference label classes: `59`
  - heldout observed label classes: `57`

### 2) How many benchmark settings were tested

- total settings tested for `scanvi`: `3`
- sweep output root:
  - `/tmp/atlasmtl_benchmarks/2026-03-05/scanvi_epoch_sweep_hlca_train10k_test5k/`
- summary:
  - `/tmp/atlasmtl_benchmarks/2026-03-05/scanvi_epoch_sweep_hlca_train10k_test5k/summary.json`
- tested configs:
  - `20/20/20` (`scvi_max_epochs/scanvi_max_epochs/query_max_epochs`)
  - `15/15/10`
  - `10/10/5`
- measured metrics:
  - label performance: `accuracy`, `macro_f1`
  - runtime: `train_elapsed_s`, `predict_elapsed_s`

### 3) Applicability to larger/more complex datasets

- this sweep is a **single-dataset local calibration**, not a universal optimum.
- what is transferable:
  - the method for choosing defaults (jointly evaluating quality + runtime).
  - the selected `15/15/10` point as a practical default for third-wave expansion.
- what still needs validation on larger/complex datasets:
  - rare-label robustness (`macro_f1` sensitivity may differ by dataset imbalance).
  - runtime scaling curve under larger `n_obs` and higher heterogeneity.
  - GPU-memory pressure on bigger references (even if current run is stable).
- operational recommendation for third-wave expansion:
  - use `15/15/10` as locked starting default.
  - for each new dataset (`PHMap`, `DISCO`, `mTCA`, `Vento`), run one quick
    confirmation slice and keep the same default unless quality drops beyond
    pre-agreed tolerance.

## Follow-up

- this pilot template is ready to be reused for the remaining third-wave
  datasets (`PHMap`, `DISCO`, `mTCA`, `Vento`) with the same CPU/GPU split policy.
- for any HLCA rerun intended to match the locked formal defaults, use the
  `*_v2.yaml` manifests instead of the historical `*_v1.yaml` manifests.
