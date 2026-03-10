# AtlasMTL Training Default Summary (`2026-03-09`)

This note records the two most recent AtlasMTL training-configuration rounds
that determine the current default method description for the paper.

## Round 1: parameter-lock benchmark (`2026-03-07`)

Purpose:

- lock a reproducible AtlasMTL training skeleton before any optimizer-default
  refinement

Scope:

- `atlasmtl` only
- CPU and GPU lock tracks
- fixed binary input path
- fixed epoch and early-stopping regime
- no promotion of extra domain/topology/calibration knobs into the locked
  benchmark baseline

Main outcome:

- the benchmark-facing AtlasMTL training backbone was locked before optimizer
  refinement
- locked benchmark defaults exported:
  - CPU: `c5_lr3e4_h256_128_b128`
  - GPU: `g6_lr1e3_h1024_512_b512`

Interpretation:

- this round should be cited as the reproducibility-lock step
- it established the stable MLP-scale training envelope on top of which later
  low-cost optimizer refinement was evaluated

Primary records:

- `documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/results_summary/atlasmtl_lock_decision_2026-03-07.md`
- `documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/results_summary/atlasmtl_locked_defaults.json`

## Round 2: low-cost optimization (`2026-03-09`)

Purpose:

- test whether a minimal optimizer-level change is sufficient to justify a new
  AtlasMTL default without reopening the entire formal benchmark stack

Stage A result:

- `AdamW + wd=5e-5` was selected as the only credible candidate
- `ReduceLROnPlateau` was screened and rejected
- no further `weight_decay` expansion was pursued after screening

Stage B design:

- compare only `baseline` vs `AdamW + wd=5e-5`
- datasets: `HLCA_Core`, `PHMap_Lung_Full_v43_light`, `mTCA`,
  `DISCO_hPBMCs`
- representative points: `build_100000_eval10k`,
  `predict_100000_10000`

Stage B interpretation:

- CPU evidence was mixed and collected under `joblib_serial_fallback`
- GPU evidence was non-degraded and treated as primary confirmation
- on Stage B GPU representative points, the candidate improved `macro_f1` on
  `7/8` points
- the only formal regression was
  `PHMap_Lung_Full_v43_light / gpu / predict_100000_10000`
- no meaningful GPU memory or RSS penalty was observed

Main decision:

- promote `AdamW + weight_decay=5e-5` as the new default training
  configuration
- do not promote `ReduceLROnPlateau`

Primary records:

- `documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/results_summary/stage_a_decision_note.md`
- `documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/results_summary/stage_b_confirmation_results.csv`
- `documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/results_summary/final_default_decision.md`

## Current paper-facing training default

The current AtlasMTL default training path should be described as:

- `input_transform="binary"`
- `optimizer_name="adamw"`
- `weight_decay=5e-5`
- `scheduler_name=None`
- `reference_storage="external"`

## Fifth-round formal refresh (`2026-03-09`)

Purpose:

- test whether the promoted software default is strong enough to replace the
  retained third-wave manuscript-grade AtlasMTL rows under the unchanged formal
  benchmark contract

Scope:

- AtlasMTL only
- main-panel rows: `16`
- supplementary `Vento` rows: `4`
- refreshed train config fixed to `AdamW + weight_decay=5e-5`

Main outcome:

- the fifth-round formal refresh does not justify replacing the retained
  third-wave manuscript-grade AtlasMTL baseline rows
- across the `16` main-panel rows, mean `delta_macro_f1 = -0.005250`
- GPU headline improvements are only `4/8`
- there is a substantial regression at
  `DISCO_hPBMCs / gpu / predict_100000_10000` with `delta_macro_f1 = -0.054224`

Interpretation:

- `AdamW + weight_decay=5e-5` remains the software default because it is still
  acceptable as a lightweight training default
- formal manuscript-grade comparison tables should continue to use the
  retained third-wave AtlasMTL baseline rows
- software default promotion and manuscript-table replacement should therefore
  be treated as separate decisions in the paper record

Primary records:

- `documents/experiments/2026-03-09_formal_atlasmtl_refresh/results_summary/formal_refresh_results.csv`
- `documents/experiments/2026-03-09_formal_atlasmtl_refresh/results_summary/formal_refresh_decision.md`
- `documents/experiments/2026-03-09_formal_atlasmtl_refresh/results_summary/implementation_record_2026-03-09.md`

## Reporting caveat

When summarizing the promotion decision, explicitly state:

- CPU optimization-round evidence was supportive but non-decisive because of
  restricted-environment `joblib_serial_fallback`
- the primary promotion basis was the clean Stage B GPU confirmation
- the `PHMap_Lung_Full_v43_light / gpu / predict_100000_10000` regression
  remains a documented caveat and should not be hidden
- the later fifth-round formal refresh did not clear the manuscript-table
  replacement bar, so formal retained AtlasMTL rows and software defaults
  should not be conflated
