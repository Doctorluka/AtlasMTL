# AtlasMTL parameter-lock benchmark dossier (`2026-03-07`)

This dossier stores the pre-formal parameter confirmation benchmark for
`atlasmtl` with separate CPU/GPU lock tracks.

## Positioning

- scope: `experiment` (pre-formal)
- objective: lock reproducible atlasmtl defaults for CPU and GPU tracks
- method: `atlasmtl` only

## Fixed run policy

- `num_threads=8` for atlasmtl benchmark runs
- `max_epochs=50` with early stopping
- `input_transform=binary` (float path already validated as inferior in prior work)
- enhanced knobs (`domain/topology/calibration`) are not part of this lock run

## Structure

- `configs/`
  - dataset registry and CPU/GPU core grids
- `scripts/`
  - input materialization, stage execution, aggregation, table rendering
- `results_summary/`
  - experiment report, execution record, lock decision, parameter guide

## Runtime roots

- prepared inputs:
  - `/tmp/atlasmtl_benchmarks/2026-03-07/reference_heldout/<dataset>/prepared/param_lock_train10k/`
- stage A:
  - `/tmp/atlasmtl_benchmarks/2026-03-07/atlasmtl_param_lock/stage_a/`
- stage B:
  - `/tmp/atlasmtl_benchmarks/2026-03-07/atlasmtl_param_lock/stage_b/`

## Reports

- primary experiment report:
  - `results_summary/experiment_report_2026-03-07_atlasmtl_param_lock.md`
- lock decision:
  - `results_summary/atlasmtl_lock_decision_2026-03-07.md`
- parameter guide:
  - `results_summary/parameter_guide_2026-03-07_atlasmtl.md`

## Current completion status (`2026-03-06`)

- stage A completed:
  - CPU `60/60` success
  - GPU `90/90` success
- stage B completed:
  - CPU `40/40` success
  - GPU `40/40` success
- lock defaults exported:
  - CPU: `c5_lr3e4_h256_128_b128`
  - GPU: `g6_lr1e3_h1024_512_b512`

## Execution note

- sandbox execution can be used for code-path smoke checks only
- official GPU evidence runs must execute in non-sandbox shell where CUDA is visible

## Quick start

```bash
python documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/scripts/prepare_atlasmtl_param_lock_inputs.py
python documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/scripts/run_atlasmtl_param_sweep.py --stage stage_a --device cpu
python documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/scripts/run_atlasmtl_param_sweep.py --stage stage_a --device cuda
python documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/scripts/aggregate_atlasmtl_param_sweep.py
python documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/scripts/run_atlasmtl_param_sweep.py --stage stage_b --device cpu
python documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/scripts/run_atlasmtl_param_sweep.py --stage stage_b --device cuda
python documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/scripts/aggregate_atlasmtl_param_sweep.py
python documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/scripts/render_atlasmtl_param_tables.py
```
