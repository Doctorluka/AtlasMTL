# Formal AtlasMTL Refresh (`2026-03-09`)

This dossier records the fifth-round formal refresh that reruns only
`atlasmtl` after the promoted default training configuration was switched to
`AdamW + weight_decay=5e-5`.

Status:

- `completed`
- formal decision: `keep_formal_atlasmtl_baseline_rows`
- recommended wording: software default retained, manuscript-grade AtlasMTL
  rows not replaced

## Scope

This round is not a full benchmark rerun.

It keeps the formal third-wave benchmark contract fixed and only refreshes the
`atlasmtl` rows for:

- main panel:
  - `HLCA_Core`
  - `PHMap_Lung_Full_v43_light`
  - `mTCA`
  - `DISCO_hPBMCs`
- supplementary:
  - `Vento`

Representative points:

- main panel:
  - `build_100000_eval10k`
  - `predict_100000_10000`
- supplementary:
  - `build_50000_eval10k`
  - `predict_50000_10000`

Tracks:

- `cpu_core`
- `gpu`

## Fixed baseline sources

- `documents/experiments/2026-03-06_formal_third_wave_scaling/results_summary/formal_main_text_snapshot_2026-03-09.csv`
- `documents/experiments/2026-03-06_formal_third_wave_scaling/results_summary/formal_comparative_performance_snapshot_2026-03-09.csv`
- `documents/experiments/2026-03-06_formal_third_wave_scaling/results_summary/formal_comparative_resource_snapshot_2026-03-09.csv`
- `documents/experiments/2026-03-06_formal_third_wave_scaling/results_summary/formal_result_scope_map_2026-03-09.csv`

## Refresh train config

- `optimizer_name: adamw`
- `weight_decay: 5e-5`
- `scheduler_name: null`

All other formal benchmark contract elements remain unchanged.

## Record locations

Repo-tracked:

- `documents/experiments/2026-03-09_formal_atlasmtl_refresh/results_summary/`

Expected result files:

- `results_summary/atlasmtl_formal_baseline_anchor.csv`
- `results_summary/formal_refresh_results.csv`
- `results_summary/formal_refresh_results.md`
- `results_summary/formal_refresh_decision.md`
- `results_summary/implementation_record_2026-03-09.md`

Tmp runtime root:

- `/tmp/atlasmtl_benchmarks/2026-03-09/formal_atlasmtl_refresh/`

## Scripts

- `scripts/freeze_formal_atlasmtl_baseline.py`
- `scripts/generate_formal_refresh_manifests.py`
- `scripts/collect_formal_refresh_results.py`
- `scripts/run_formal_refresh_cpu.sh`
- `scripts/run_formal_refresh_gpu.sh`

## Outcome summary

The refresh completed on all `20` planned AtlasMTL-only points.

Main-panel outcome:

- `16/16` rows completed
- mean `delta_macro_f1 = -0.005250`
- GPU headline improvements: `4/8`
- worst main-panel regression: `DISCO_hPBMCs / gpu / predict_100000_10000`
  with `delta_macro_f1 = -0.054224`
- GPU median train memory delta remained small: `0.003550 GB`

Decision:

- keep the retained formal third-wave AtlasMTL baseline rows in manuscript-grade
  formal comparison tables
- keep the promoted code default as `AdamW + weight_decay=5e-5`
- document this fifth-round refresh as a non-promotion for formal row
  replacement

Recommended formal expression:

The fifth-round formal refresh does not justify replacing the retained
third-wave manuscript-grade AtlasMTL baseline rows. The refreshed
configuration (`AdamW + weight_decay=5e-5`) remains the software default
because it is still acceptable as a lightweight training default, but the
formal refresh evidence is not strong enough for manuscript-table replacement:
across the `16` main-panel rows, the mean `delta_macro_f1` is `-0.005250`, GPU
headline improvements are only `4/8`, and there is a substantial regression at
`DISCO_hPBMCs / gpu / predict_100000_10000` (`-0.054224`). Therefore, formal
comparison tables should continue to use the retained third-wave AtlasMTL
baseline rows.
