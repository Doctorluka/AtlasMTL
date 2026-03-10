# 2026-03-10 Weight Activation Rule Validation

## Objective

Validate a first-pass, error-driven activation rule for AtlasMTL task
weighting.

The immediate question is not "what is the globally best task-weight vector?".
It is:

- should a new dataset stay on `uniform` task weights
- or should it enter a small non-uniform weighting candidate test

This dossier uses the two completed `study`-split cases as the minimum
positive/negative contrast, and one additional shallower sanity-check case:

- `PHMap_Lung_Full_v43_light`: should activate non-uniform weighting
- `HLCA_Core`: should stay on `uniform`
- `mTCA`: should stay on `uniform`

## Rule framing

`activation_rule_v1` is intentionally conservative.

It triggers non-uniform weighting only when both are true:

1. the baseline shows clear fine-level difficulty
2. the baseline also shows a meaningful hierarchy-structured tradeoff

The rule is designed as a framework policy helper, not a black-box searcher.

## Outputs

- `results_summary/weight_activation_features.csv`
- `results_summary/weight_activation_decisions.json`
- `results_summary/weight_activation_paper_table.csv`
- `results_summary/weight_activation_summary.md`

## Current result

- `PH-Map`: `activate_nonuniform_weighting = true`
- `HLCA`: `activate_nonuniform_weighting = false`
- `mTCA`: `activate_nonuniform_weighting = false`

This is sufficient for a first framework-level policy claim:

non-uniform task weighting should be treated as an error-driven,
dataset-adaptive option rather than a universal default.
