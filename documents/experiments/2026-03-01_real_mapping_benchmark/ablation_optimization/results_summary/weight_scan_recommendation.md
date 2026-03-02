# Weight Scan Recommendation

This note records the current operational recommendation from the completed
task-weight search. It is a benchmark-internal recommendation, not a
project-wide invariant.

## Rule

Recommendations were selected by:

- computing `quality_score = 0.5 * lv4_accuracy + 0.5 * lv4_macro_f1`
- keeping runs within `95%` of the best score for each device
- excluding major coarse/full-path regression when possible
- choosing the lowest-resource candidate inside the eligible near-optimal band

## Current recommendation

- `cpu`: `ratio_1.6`
- `cuda`: `lv4strong_a = [0.2, 0.7, 1.5, 3.0]`

## Interpretation

- the current sampled dataset supports stronger fine-level emphasis
- `anno_lv4` weight of `3.0` is a valid candidate and should remain in the next
  benchmark design
- the final project default should still be validated on at least one
  additional dataset before being frozen
