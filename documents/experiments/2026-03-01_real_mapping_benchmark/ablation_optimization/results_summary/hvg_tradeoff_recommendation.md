# HVG Tradeoff Recommendation

This note records the current operational recommendation from the completed HVG
tradeoff run. It is a benchmark-internal recommendation, not a project-wide
invariant.

## Rule

Recommendations were selected by:

- computing `quality_score = 0.5 * lv4_accuracy + 0.5 * lv4_macro_f1`
- keeping runs within `95%` of the best score for each device
- choosing the lowest-resource candidate inside that near-optimal band

## Current recommendation

- `cpu`: use `hvg5000`
- `cuda`: use `hvg6000`

## Interpretation

- `cpu` and `cuda` should remain separate AtlasMTL variants in benchmark
  tables
- `whole` should remain in the search grid as the stability baseline
- the current recommendation is dataset-level, not universal
