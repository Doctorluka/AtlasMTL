# HLCA formal third-wave benchmark record (`train10k`, `test5k`)

## Run metadata

- date: `2026-03-05`
- operator: `codex`
- environment: `atlasmtl-env`

## Manifest references

- CPU manifest:
  - `documents/experiments/2026-03-05_formal_third_wave_hlca_benchmark/manifests/reference_heldout/HLCA_Core__ann_level_5__formal_runtime_cpu_train10k_test5k_v1.yaml`
- GPU manifest:
  - `documents/experiments/2026-03-05_formal_third_wave_hlca_benchmark/manifests/reference_heldout/HLCA_Core__ann_level_5__formal_runtime_gpu_train10k_test5k_v1.yaml`

## Commands

- split materialization:
  - `python documents/experiments/2026-03-05_formal_third_wave_hlca_benchmark/scripts/prepare_formal_hlca_train10k_test5k.py`
- environment snapshot:
  - `bash documents/experiments/2026-03-05_formal_third_wave_hlca_benchmark/scripts/record_environment_versions.sh`
- CPU group:
  - `bash documents/experiments/2026-03-05_formal_third_wave_hlca_benchmark/scripts/run_formal_hlca_cpu_group.sh`
- GPU group:
  - `bash documents/experiments/2026-03-05_formal_third_wave_hlca_benchmark/scripts/run_formal_hlca_gpu_group.sh`

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

## Follow-up

- this pilot template is ready to be reused for the remaining third-wave
  datasets (`PHMap`, `DISCO`, `mTCA`, `Vento`) with the same CPU/GPU split policy.
