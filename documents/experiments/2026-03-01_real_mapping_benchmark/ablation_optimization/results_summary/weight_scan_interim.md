# Weight Scan Interim Summary

This note records the completed task-weight search for the current benchmark
optimization task. It is an internal decision document, not a final paper
result.

## Scope

This run fixed:

- feature space = `hvg6000`
- `input_transform = binary`
- counts source = `layers["counts"]`
- hierarchy enabled
- KNN correction off

This run compared:

- anchor weights:
  - `uniform`
  - `phmap`
  - `lv4strong_a = [0.2, 0.7, 1.5, 3.0]`
  - `lv4strong_b = [0.3, 0.8, 2.0, 3.0]`
- ratio scan:
  - `r = 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0`

for both `cpu` and `cuda`.

## Current recommendation

- `cpu`: `ratio_1.6`
  - weights:
    `[0.43215211754537597, 0.6914433880726015, 1.1063094209161626, 1.7700950734658603]`
  - `lv4_accuracy = 0.7653`
  - `lv4_macro_f1 = 0.6548`
  - `full_path_accuracy = 0.7580`
  - `peak_rss_gb = 3.4174`
  - `train_elapsed_seconds = 10.0675`
- `cuda`: `lv4strong_a`
  - weights: `[0.2, 0.7, 1.5, 3.0]`
  - `lv4_accuracy = 0.7753`
  - `lv4_macro_f1 = 0.6819`
  - `full_path_accuracy = 0.7660`
  - `peak_rss_gb = 3.4169`
  - `train_elapsed_seconds = 2.6873`

## Reading

- stronger fine-level weighting is worthwhile on the current sampled dataset
- `anno_lv4` weight reaching `3.0` is not only feasible, but currently gives
  the best `cuda` recommendation
- the `cpu` recommendation is more moderate and prefers a ratio-scan solution
  instead of the strongest anchor
- the best weight schedule is therefore likely device-sensitive and potentially
  dataset-sensitive

## Runtime artifacts

- `~/tmp/atlasmtl_real_mapping_benchmark_20260301/ablation_optimization/weight_scan/metrics.json`
- `~/tmp/atlasmtl_real_mapping_benchmark_20260301/ablation_optimization/weight_scan/analysis/atlasmtl_weight_scan.csv`
- `~/tmp/atlasmtl_real_mapping_benchmark_20260301/ablation_optimization/weight_scan/analysis/atlasmtl_weight_recommendations.csv`
- `~/tmp/atlasmtl_real_mapping_benchmark_20260301/ablation_optimization/weight_scan/analysis/weight_scan_recommendation.md`
- `~/tmp/atlasmtl_real_mapping_benchmark_20260301/ablation_optimization/weight_scan/analysis/figures/`

## Boundary

This summary is still interim:

- it is based on the current sampled dataset only
- it should guide the next benchmark round
- it should not be promoted directly to the final paper conclusion without
  cross-dataset confirmation
