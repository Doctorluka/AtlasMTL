# HLCA formal third-wave error/fix note (`train10k`, `test5k`)

Use this note for any failed run attempt and correction cycle.

## Entry template

- timestamp:
- group: `cpu` or `gpu`
- method:
- failure symptom:
- log path:
- root cause:
- fix applied:
- rerun command:
- rerun result:

## 2026-03-05 entry #1

- timestamp: `2026-03-05`
- group: `gpu`
- method: `atlasmtl` and `scanvi`
- failure symptom:
  - `atlasmtl`: `device='cuda' was requested but CUDA is not available`
  - `scanvi`: CUDA init/NVML unavailable, then monitor call fails at
    `reset_peak_memory_stats`
- log path:
  - `/tmp/atlasmtl_benchmarks/2026-03-05/reference_heldout/HLCA_Core/benchmark/formal_train10k_test5k/gpu_group_v1/runs/atlasmtl/stderr.log`
  - `/tmp/atlasmtl_benchmarks/2026-03-05/reference_heldout/HLCA_Core/benchmark/formal_train10k_test5k/gpu_group_v1/runs/scanvi/stderr.log`
- root cause:
  - current execution environment has no usable CUDA device
- fix applied:
  - none in this environment; marked as infrastructure blocker
- rerun command:
  - `bash documents/experiments/2026-03-05_formal_third_wave_hlca_benchmark/scripts/run_formal_hlca_gpu_group.sh`
- rerun result:
  - pending (requires CUDA-enabled host)

## 2026-03-05 entry #2

- timestamp: `2026-03-05`
- group: `gpu`
- method: `atlasmtl` and `scanvi`
- failure symptom:
  - none on rerun
- log path:
  - `/tmp/atlasmtl_benchmarks/2026-03-05/reference_heldout/HLCA_Core/benchmark/formal_train10k_test5k/gpu_group_v1/runs/atlasmtl/`
  - `/tmp/atlasmtl_benchmarks/2026-03-05/reference_heldout/HLCA_Core/benchmark/formal_train10k_test5k/gpu_group_v1/runs/scanvi/`
- root cause:
  - previous failure was infrastructure/sandbox visibility issue, not method logic
- fix applied:
  - rerun `run_formal_hlca_gpu_group.sh` outside restricted sandbox
- rerun command:
  - `bash documents/experiments/2026-03-05_formal_third_wave_hlca_benchmark/scripts/run_formal_hlca_gpu_group.sh`
- rerun result:
  - success (`2/2`)
