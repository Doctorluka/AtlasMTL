# Formal third-wave HLCA plan (`2026-03-05`)

This plan defines the formal third-wave pilot run for `HLCA_Core` with:

- train model on `10k` reference cells
- predict on `5k` heldout cells
- dual execution groups:
  - `cpu_group`: all methods except `scanvi`
  - `gpu_group`: `atlasmtl` + `scanvi`

## Locked scope

- dataset: `HLCA_Core`
- label level: `ann_level_5`
- split key: `donor_id`
- counts contract: `layers["counts"]`

## Runtime fairness contract

- fairness policy: `cpu_only_strict` for CPU group table
- GPU group reported separately
- fixed thread env vars for both groups:
  - `OMP_NUM_THREADS=8`
  - `MKL_NUM_THREADS=8`
  - `OPENBLAS_NUM_THREADS=8`
  - `NUMEXPR_NUM_THREADS=8`
- record degraded mode if `joblib` falls back to serial.

## Output structure

Repo dossier:

- `documents/experiments/2026-03-05_formal_third_wave_hlca_benchmark/`

Runtime roots:

- `/tmp/atlasmtl_benchmarks/2026-03-05/reference_heldout/HLCA_Core/prepared/formal_train10k_test5k/`
- `/tmp/atlasmtl_benchmarks/2026-03-05/reference_heldout/HLCA_Core/benchmark/formal_train10k_test5k/cpu_group_v1/`
- `/tmp/atlasmtl_benchmarks/2026-03-05/reference_heldout/HLCA_Core/benchmark/formal_train10k_test5k/gpu_group_v1/`

## Acceptance checks

- both manifests resolve valid local files
- CPU group excludes `scanvi`
- GPU group contains only `atlasmtl` and `scanvi`
- environment snapshots are written before benchmark execution
- each run emits `scaleout_status.json`, `summary.csv`, and `metrics.json`.
