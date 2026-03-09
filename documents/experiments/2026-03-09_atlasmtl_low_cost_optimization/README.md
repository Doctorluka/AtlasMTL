# AtlasMTL Low-Cost Optimization (`2026-03-09`)

This dossier records the fourth-round atlasmtl optimization work.

The purpose of this round is narrow and explicit:

- test whether a minimal training upgrade improves `atlasmtl` enough to justify
  a new default
- avoid reopening the completed third-wave formal benchmark unless a later
  targeted rerun is genuinely needed

## Scope

Primary target:

- low-cost training improvements inside `atlasmtl`

Locked candidate changes:

- `AdamW`
- `weight_decay`
- `ReduceLROnPlateau`

Out of scope for this dossier:

- full benchmark redesign
- new comparator methods
- multi-level hierarchy benchmark rerun
- new model architecture branches

## Baseline context

This round starts **after**:

- the `2026-03-05` HLCA formal pilot
- the `2026-03-06` third-wave formal scaling round
- the `2026-03-07` atlasmtl parameter-lock round

Those rounds already established:

- the benchmark execution template
- the formal split/preprocessing contract
- the current locked atlasmtl defaults
- the main performance/resource evidence base

This round therefore focuses on framework refinement, not benchmark expansion.

## Main reference files

Use these files as the fixed background for this dossier:

- `plan/2026-03-09_fourth_round_atlasmtl_optimization_plan.md`
- `plan/2026-03-06_formal_third_wave_scaling_plan.md`
- `documents/protocols/formal_third_wave_scaling_protocol.md`
- `documents/experiments/2026-03-06_formal_third_wave_scaling/results_summary/formal_experiment_report.md`
- `documents/experiments/2026-03-06_formal_third_wave_scaling/results_summary/formal_main_text_snapshot_2026-03-09.csv`
- `documents/experiments/2026-03-05_formal_third_wave_hlca_benchmark/results_summary/round_summary_2026-03-05_formal_hlca.md`

## Execution design

This round uses two stages.

### Stage A: screening

Datasets:

- `PHMap_Lung_Full_v43_light`
- `mTCA`

Representative benchmark points:

- `build_100000_eval10k`
- `predict_100000_10000`

Configurations:

1. baseline
2. `AdamW + wd=1e-5`
3. `AdamW + wd=5e-5`
4. `AdamW + wd=1e-4`
5. best `wd` + `ReduceLROnPlateau`

### Stage B: confirmation

Only run if Stage A identifies a credible candidate default.

Datasets:

- `HLCA_Core`
- `PHMap_Lung_Full_v43_light`
- `mTCA`
- `DISCO_hPBMCs`

Representative points:

- `build_100000_eval10k`
- `predict_100000_10000`

Configurations:

- old baseline
- new candidate default

## Record locations

### Repo-tracked dossier root

- `documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/`

### Repo-tracked summaries

- `documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/results_summary/`

Recommended files to create during execution:

- `results_summary/stage_a_screening_results.csv`
- `results_summary/stage_a_screening_results.md`
- `results_summary/stage_a_decision_note.md`
- `results_summary/stage_b_confirmation_results.csv`
- `results_summary/stage_b_confirmation_results.md`
- `results_summary/final_default_decision.md`

### Tmp runtime root

- `/tmp/atlasmtl_benchmarks/2026-03-09/atlasmtl_low_cost_optimization/`

Recommended tmp structure:

- `/tmp/atlasmtl_benchmarks/2026-03-09/atlasmtl_low_cost_optimization/<dataset>/<track>/<point>/<config_name>/`

Each run folder should retain:

- `metrics.json`
- `summary.csv`
- `stdout.log`
- `stderr.log`
- any model artifacts needed for audit

## Decision rule

Promote a new default only if it satisfies all of the following:

- no meaningful regression on `mTCA`
- neutral-to-better `macro_f1` on `PHMap`
- no meaningful runtime penalty
- no meaningful peak-memory penalty
- signs of improved convergence stability

If these conditions are not met, keep the current baseline and stop.

## Benchmark rerun policy

Default rule:

- do **not** rerun the full third-wave formal benchmark

Rerun only if:

- a new default clearly survives Stage B and representative-point confirmation
  is needed, or
- a later hierarchy secondary analysis requires per-cell predictions that were
  not retained before

## Current status

Current status: `planning only`

No execution should begin until:

- this dossier is reviewed
- the plan file is approved
- the exact result tables to be written are accepted
