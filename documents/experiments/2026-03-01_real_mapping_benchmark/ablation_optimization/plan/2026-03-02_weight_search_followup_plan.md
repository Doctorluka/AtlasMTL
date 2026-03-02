# AtlasMTL Weight Search Follow-up Plan

## Goal

The next weight-search round should test whether stronger fine-level emphasis
improves `anno_lv4` without causing unacceptable regression in coarse levels or
full-path quality.

## Locked decision rule

- primary quality targets:
  - `anno_lv4` accuracy
  - `anno_lv4` macro-F1
- required guardrails:
  - `anno_lv1` and `anno_lv2` should not regress materially
  - `full_path_accuracy` and `full_path_coverage` should remain stable
- required resource targets:
  - train elapsed seconds
  - predict elapsed seconds
  - peak RSS
  - peak GPU memory when applicable

## Search scope

Keep fixed:

- counts source: `layers["counts"]`
- input transform: `binary`
- feature space:
  - preferred: dataset-level HVG recommendation from the HVG tradeoff line
  - temporary fallback: `hvg6000`
- hierarchy: enabled
- KNN correction: disabled for this dataset

Vary:

- anchor weights:
  - `uniform = [1.0, 1.0, 1.0, 1.0]`
  - `phmap = [0.3, 0.8, 1.5, 2.0]`
  - `lv4strong_a = [0.2, 0.7, 1.5, 3.0]`
  - `lv4strong_b = [0.3, 0.8, 2.0, 3.0]`
- parameterized ratio scan:
  - generate `[1, r, r^2, r^3]`
  - normalize to mean 1
  - scan `r = 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0`

## Outputs

- `atlasmtl_weight_scan.csv`
- `atlasmtl_weight_recommendations.csv`
- `weight_scan_interim.md`
- `weight_scan_recommendation.md`

## Success criteria

- the project can justify whether `anno_lv4` should receive stronger weighting
- the recommendation is based on both fine-level gain and coarse/full-path
  stability
- any coarse-level regression remains explicit in the summary
