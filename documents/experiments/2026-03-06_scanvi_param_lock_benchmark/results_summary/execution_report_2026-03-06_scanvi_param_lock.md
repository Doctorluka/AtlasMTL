# Scanvi param-lock benchmark execution report (`2026-03-06`)

## Run metadata

- date: `2026-03-06`
- scope: `experiment` (pre-formal parameter confirmation)
- method: `scanvi` only
- runtime mode: `GPU only`

## Stages and status

- stage A (coarse grid):
  - expected runs: `40` (`5 datasets x 8 params x 1 seed x 1 query size`)
  - completed: `40`
  - success: `40`
  - failure: `0`
- stage B (top-2 confirmation):
  - expected runs: `40` (`5 datasets x 2 params x 2 seeds x 2 query sizes`)
  - completed: `40`
  - success: `40`
  - failure: `0`

## Data coverage

- `PHMap_Lung_Full_v43_light`
- `DISCO_hPBMCs`
- `mTCA`
- `HLCA_Core`
- `Vento`

All datasets completed full stage-A and stage-B schedules.

## Core outputs

- stage-A run index:
  - `/tmp/atlasmtl_benchmarks/2026-03-06/scanvi_param_lock/stage_a/run_index.csv`
- stage-B run index:
  - `/tmp/atlasmtl_benchmarks/2026-03-06/scanvi_param_lock/stage_b/run_index.csv`
- aggregated raw:
  - `documents/experiments/2026-03-06_scanvi_param_lock_benchmark/results_summary/sweep_raw_results.csv`
- stage-A ranking:
  - `documents/experiments/2026-03-06_scanvi_param_lock_benchmark/results_summary/stage_a_param_ranking.csv`
- stage-B stability:
  - `documents/experiments/2026-03-06_scanvi_param_lock_benchmark/results_summary/stage_b_stability.csv`
- selected top-2 parameters:
  - `documents/experiments/2026-03-06_scanvi_param_lock_benchmark/results_summary/top2_params.json`

## Top-2 parameters selected from stage A

1. `p8_e25_25_20_nl20`
   - `scvi_max_epochs=25`
   - `scanvi_max_epochs=25`
   - `query_max_epochs=20`
   - `n_latent=20`
2. `p6_e20_20_20_nl20`
   - `scvi_max_epochs=20`
   - `scanvi_max_epochs=20`
   - `query_max_epochs=20`
   - `n_latent=20`

## Notes

- This run excludes `scanvi` CPU benchmarking by design.
- GPU execution must be validated in non-sandbox shell for final evidence runs.
