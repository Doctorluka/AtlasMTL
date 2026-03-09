# Execution report: HLCA formal main batch start

Date: `2026-03-06`

This report records the start of the first formal main-panel execution batch for
`HLCA_Core` after the formal plan review and sanity-check pass.

## Plan review outcome

The formal plan is confirmed as executable with one operational grouping rule:

- GPU track runs `atlasmtl` and `scanvi`
- CPU core track runs `atlasmtl`, `reference_knn`, `celltypist`, `singler`,
  and `symphony`
- CPU `seurat_anchor_transfer` runs as a separate isolated track

Reason:

- `seurat_anchor_transfer` is a known long-runtime CPU comparator on
  `HLCA 100k -> 10k`
- the round now carries a locked `60-minute` stop rule for single-method runs

## Batch scripts

- `scripts/run_formal_hlca_gpu_main_batch.sh`
- `scripts/run_formal_hlca_cpu_core_main_batch.sh`
- `scripts/run_formal_hlca_cpu_seurat_main_batch.sh`

## Planned output roots

- GPU:
  - `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/HLCA_Core/benchmark/formal_main/gpu/`
- CPU core:
  - `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/HLCA_Core/benchmark/formal_main/cpu_core/`
- CPU seurat:
  - `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/HLCA_Core/benchmark/formal_main/cpu_seurat/`

## Track scope

GPU main track:

- build scaling:
  - `10k / 20k / 30k / 50k / 100k / 150k / 200k / 300k`
- predict scaling:
  - `1k / 3k / 5k / 8k / 10k / 15k / 20k / 50k`

CPU core main track:

- build scaling:
  - `10k / 20k / 30k / 50k / 100k / 150k / 200k / 300k`
- predict scaling:
  - `1k / 3k / 5k / 8k / 10k / 15k / 20k / 50k`

CPU seurat isolated track:

- same manifest grid as CPU core
- each manifest run is guarded by `timeout=3600s`
- timeout events are recorded as `manual_stop_long_runtime`

## Launch update

Observed launch state after start:

- `cpu_core` started successfully and entered `build_10000_eval10k`
- `cpu_seurat` started successfully and entered `build_10000_eval10k`
- `gpu` did not start in the current execution context because
  `torch.cuda.is_available()` was `false` during preflight

Interpretation:

- the GPU non-start here is an execution-environment constraint
- it is not recorded as a formal method failure

Follow-up handling:

- the `HLCA` formal GPU main batch was restarted from a direct non-sandbox shell
  session
- that direct-shell launch passed CUDA preflight on `NVIDIA GeForce RTX 4090`
- this handling rule is now part of the formal protocol for GPU execution
