# Fifth-Round Formal AtlasMTL Refresh Plan (`2026-03-09`)

This round refreshes only the formal `atlasmtl` rows after the default training
configuration was promoted to `AdamW + weight_decay=5e-5`.

## Goal

- rerun only `atlasmtl` under the same formal third-wave benchmark contract
- compare the new results against the retained formal `atlasmtl` baseline
- decide whether the formal manuscript-facing `atlasmtl` rows should be replaced

## Fixed comparison scope

Main panel (`16` points):

- `HLCA_Core`
- `PHMap_Lung_Full_v43_light`
- `mTCA`
- `DISCO_hPBMCs`
- tracks: `cpu_core`, `gpu`
- points: `build_100000_eval10k`, `predict_100000_10000`

Supplementary (`4` points):

- `Vento`
- tracks: `cpu_core`, `gpu`
- points: `build_50000_eval10k`, `predict_50000_10000`

## Frozen baseline sources

- `documents/experiments/2026-03-06_formal_third_wave_scaling/results_summary/formal_comparative_performance_snapshot_2026-03-09.csv`
- `documents/experiments/2026-03-06_formal_third_wave_scaling/results_summary/formal_comparative_resource_snapshot_2026-03-09.csv`
- `documents/experiments/2026-03-06_formal_third_wave_scaling/results_summary/formal_result_scope_map_2026-03-09.csv`

## Refresh contract

Keep unchanged:

- dataset manifests and split definitions
- preprocessing contract
- build/predict sizes
- device-group separation
- fairness-policy labeling
- comparator rows and retained formal snapshots

Replace only:

- `atlasmtl` training config
  - `optimizer_name: adamw`
  - `weight_decay: 5e-5`
  - `scheduler_name: null`

## Expected outputs

Repo-tracked dossier root:

- `documents/experiments/2026-03-09_formal_atlasmtl_refresh/`

Key files:

- `results_summary/atlasmtl_formal_baseline_anchor.csv`
- `results_summary/formal_refresh_results.csv`
- `results_summary/formal_refresh_results.md`
- `results_summary/formal_refresh_decision.md`

Tmp runtime root:

- `/tmp/atlasmtl_benchmarks/2026-03-09/formal_atlasmtl_refresh/`

## Decision rule

Promote refreshed formal `atlasmtl` rows if all of the following hold:

- main-panel `delta_macro_f1` mean is non-negative
- GPU main-panel points improve on at least `5/8` headline rows
- no main-panel point has `delta_macro_f1 < -0.02`
- no meaningful GPU-memory or RSS penalty appears

If these conditions fail, keep the retained formal baseline rows and report the
new default as code-level only.
