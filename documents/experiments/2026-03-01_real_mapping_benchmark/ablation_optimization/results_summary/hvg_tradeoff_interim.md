# HVG Tradeoff Interim Summary

This note records the completed HVG tradeoff run for the current benchmark
optimization task. It is an internal decision document, not a final paper
result.

## Scope

This run fixed:

- `input_transform = binary`
- `task_weights = [0.3, 0.8, 1.5, 2.0]`
- counts source = `layers["counts"]`
- hierarchy enabled
- KNN correction off

This run compared:

- `whole`
- `hvg3000`
- `hvg4000`
- `hvg5000`
- `hvg6000`
- `hvg7000`
- `hvg8000`

for both `cpu` and `cuda`.

## Current recommendation

- `cpu`: `hvg5000`
  - `lv4_accuracy = 0.7700`
  - `lv4_macro_f1 = 0.6499`
  - `full_path_accuracy = 0.7643`
  - `peak_rss_gb = 3.4166`
  - `train_elapsed_seconds = 9.7801`
- `cuda`: `hvg6000`
  - `lv4_accuracy = 0.7643`
  - `lv4_macro_f1 = 0.6586`
  - `full_path_accuracy = 0.7587`
  - `peak_rss_gb = 3.4168`
  - `train_elapsed_seconds = 2.7853`

## Reading

- the current data no longer supports a simplistic rule of always using
  `hvg6000`
- `whole` remains the required anchor baseline, but it was not the current
  recommendation under the quality-resource rule
- the preferred HVG count appears device-sensitive on this sampled dataset:
  `cpu` leans to `5000`, while `cuda` leans to `6000`

## Runtime artifacts

- `~/tmp/atlasmtl_real_mapping_benchmark_20260301/ablation_optimization/hvg_tradeoff/metrics.json`
- `~/tmp/atlasmtl_real_mapping_benchmark_20260301/ablation_optimization/hvg_tradeoff/analysis/atlasmtl_hvg_tradeoff.csv`
- `~/tmp/atlasmtl_real_mapping_benchmark_20260301/ablation_optimization/hvg_tradeoff/analysis/atlasmtl_hvg_recommendations.csv`
- `~/tmp/atlasmtl_real_mapping_benchmark_20260301/ablation_optimization/hvg_tradeoff/analysis/hvg_tradeoff_recommendation.md`
- `~/tmp/atlasmtl_real_mapping_benchmark_20260301/ablation_optimization/hvg_tradeoff/analysis/figures/`

## Boundary

This summary is still interim:

- it is based on the current sampled dataset only
- it should guide the next benchmark round
- it should not be promoted directly to the final paper conclusion without
  cross-dataset confirmation
