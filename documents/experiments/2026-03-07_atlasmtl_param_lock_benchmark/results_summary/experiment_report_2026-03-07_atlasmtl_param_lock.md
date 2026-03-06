# Experiment report: atlasmtl parameter lock benchmark (`2026-03-07`)

## Objective

Lock default atlasmtl parameters for both CPU and GPU tracks before the formal
benchmark round.

## Fixed protocol

- `num_threads=8`
- `max_epochs=50`
- `val_fraction=0.1`
- `early_stopping_patience=5`
- `input_transform=binary`
- `reference_storage=external`

## Stages

- stage A:
  - CPU core grid: `12` combos
  - GPU core grid: `18` combos
  - query size: `5k`
  - seed: `2026`
- stage B:
  - per-track top-2 from stage A
  - seeds: `17`, `23`
  - query sizes: `5k`, `10k`

## Output paths

- runtime root:
  - `/tmp/atlasmtl_benchmarks/2026-03-07/atlasmtl_param_lock/`
- aggregated outputs:
  - `stage_a_core_ranking_cpu.csv`
  - `stage_a_core_ranking_gpu.csv`
  - `stage_b_stability_cpu.csv`
  - `stage_b_stability_gpu.csv`
  - `atlasmtl_locked_defaults.json`

## Status

- [x] stage A CPU completed (`60/60`, fail=`0`)
- [x] stage A GPU completed (`90/90`, fail=`0`)
- [x] stage B CPU completed (`40/40`, fail=`0`)
- [x] stage B GPU completed (`40/40`, fail=`0`)
- [x] lock defaults exported

## Key outputs (final)

- `documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/results_summary/aggregation_summary.json`
- `documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/results_summary/atlasmtl_locked_defaults.json`
- `documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/results_summary/stage_a_core_ranking_cpu.csv`
- `documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/results_summary/stage_a_core_ranking_gpu.csv`
- `documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/results_summary/stage_b_stability_cpu.csv`
- `documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/results_summary/stage_b_stability_gpu.csv`

## Stage-A top result snapshot

- CPU top-1:
  - `param_id=c5_lr3e4_h256_128_b128`
  - `mean_macro_f1=0.6911`
  - `mean_accuracy=0.8132`
  - `mean_total_elapsed_seconds=5.8106`
- GPU top-1:
  - `param_id=g6_lr1e3_h1024_512_b512`
  - `mean_macro_f1=0.7069`
  - `mean_accuracy=0.8204`
  - `mean_total_elapsed_seconds=1.5764`

## Stage-B stability snapshot

- CPU:
  - `c5_lr3e4_h256_128_b128`: `macro_f1=0.6970 (5k) / 0.7000 (10k)`
  - `c2_lr1e3_h256_128_b256`: `macro_f1=0.6861 (5k) / 0.6898 (10k)`
- GPU:
  - `g6_lr1e3_h1024_512_b512`: `macro_f1=0.7031 (5k) / 0.6998 (10k)`
  - `g12_lr3e4_h1024_512_b512`: `macro_f1=0.6919 (5k) / 0.6893 (10k)`

## Error/fix record

- issue:
  - Stage-B launch failed initially with:
    - `ValueError: invalid literal for int() with base 10: ','`
- root cause:
  - `top2_params.json` stores `hidden_sizes` as CSV string (`"256,128"`), while
    Stage-B manifest builder assumed list-like and attempted char-wise `int()`
    conversion.
- fix:
  - updated parser in:
    - `documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/scripts/run_atlasmtl_param_sweep.py`
  - added `_parse_hidden_sizes(...)` to support list/tuple/string inputs.
- verification:
  - rerun Stage-B CPU/GPU after patch; both reached `40/40`, fail=`0`.
