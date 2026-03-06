# Experiment report: scanvi parameter lock benchmark (`2026-03-06`)

## Objective

Confirm a robust default `scanvi` parameter set for later formal experiments,
using cross-dataset and multi-setting evidence (pre-formal experiment scope).

## Setup

- method: `scanvi` only
- runtime: GPU only
- datasets:
  - `PHMap_Lung_Full_v43_light`
  - `DISCO_hPBMCs`
  - `mTCA`
  - `HLCA_Core`
  - `Vento`
- split contract:
  - train: `reference_train_10k.h5ad`
  - predict: `heldout_test_5k.h5ad` and `heldout_test_10k.h5ad`

## Stage design and completion

- stage A:
  - design: `5 datasets x 8 param combos x 1 seed x 5k`
  - expected runs: `40`
  - completed: `40`
  - success: `40`
  - failure: `0`
- stage B:
  - design: `5 datasets x top-2 params x 2 seeds x (5k,10k)`
  - expected runs: `40`
  - completed: `40`
  - success: `40`
  - failure: `0`

Total official evidence runs: `80`, all successful.

## Stage-A ranking summary

Top-ranked parameter sets (`macro_f1` primary, `accuracy` secondary):

1. `p8_e25_25_20_nl20`
   - mean `macro_f1`: `0.7459`
   - mean `accuracy`: `0.8402`
   - mean total elapsed: `41.24s`
2. `p6_e20_20_20_nl20`
   - mean `macro_f1`: `0.7261`
   - mean `accuracy`: `0.8391`
   - mean total elapsed: `34.41s`
3. `p7_e25_25_20_nl10`
   - mean `macro_f1`: `0.7231`
   - mean `accuracy`: `0.8336`

## Stage-B stability summary

Top-2 confirmation across seeds and query sizes:

- `query_size=5k`:
  - `p8` mean `macro_f1=0.7477`, mean `accuracy=0.8408`
  - `p6` mean `macro_f1=0.7310`, mean `accuracy=0.8383`
- `query_size=10k`:
  - `p8` mean `macro_f1=0.7362`, mean `accuracy=0.8385`
  - `p6` mean `macro_f1=0.7367`, mean `accuracy=0.8372`

Interpretation:

- `p8` is strongest on overall cross-dataset performance and remains stable in
  stage-B confirmation.
- `p6` is a lower-runtime backup candidate with competitive performance.

## Final lock for formal experiments

Default lock:

- `scvi_max_epochs=25`
- `scanvi_max_epochs=25`
- `query_max_epochs=20`
- `n_latent=20`
- `batch_size=256`
- `datasplitter_num_workers=0`

## Reproducible output paths

- stage A run index:
  - `/tmp/atlasmtl_benchmarks/2026-03-06/scanvi_param_lock/stage_a/run_index.csv`
- stage B run index:
  - `/tmp/atlasmtl_benchmarks/2026-03-06/scanvi_param_lock/stage_b/run_index.csv`
- aggregated raw and tables:
  - `documents/experiments/2026-03-06_scanvi_param_lock_benchmark/results_summary/sweep_raw_results.csv`
  - `documents/experiments/2026-03-06_scanvi_param_lock_benchmark/results_summary/stage_a_param_ranking.csv`
  - `documents/experiments/2026-03-06_scanvi_param_lock_benchmark/results_summary/stage_b_stability.csv`
  - `documents/experiments/2026-03-06_scanvi_param_lock_benchmark/results_summary/top2_params.json`
