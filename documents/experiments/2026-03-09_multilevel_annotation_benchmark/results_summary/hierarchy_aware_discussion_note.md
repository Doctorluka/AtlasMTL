# Hierarchy-Aware Discussion Note

## Question raised

The main follow-up question after the sixth-round results was whether
`hierarchy-aware` prediction is fully beneficial, and whether turning
`enforce_hierarchy` off might improve the weakest finest-level results.

## What the current implementation does

In the current AtlasMTL implementation, `enforce_hierarchy=True` does not change
training. It is a post-prediction consistency pass:

- per-level predictions are first produced independently
- parent-child consistency is then checked
- if a child prediction maps to a different parent than the predicted parent,
  the child prediction is forced to `Unknown`

So the main practical effect of hierarchy enforcement is:

- lower finest-level `coverage`
- possible decline in finest-level headline metrics
- fully consistent output paths

## Why turning it off may help

This round showed non-zero hierarchy enforcement rates on the finest level:

- `HLCA_Core`
  - roughly `0.89%` to `1.25%`
- `PHMap_Lung_Full_v43_light`
  - roughly `1.48%` to `4.61%`
- `DISCO_hPBMCs`
  - roughly `0.92%` to `1.11%`
- `mTCA`
  - roughly `0.50%` to `0.72%`

This implies that disabling hierarchy enforcement would probably recover some
currently rejected finest-level predictions, especially on:

- `PHMap_Lung_Full_v43_light`
- secondarily `HLCA_Core` and `DISCO_hPBMCs`

## Why turning it off may not be sufficient

Even if hierarchy enforcement is disabled, the multi-level model remains a
jointly trained multi-task model. Therefore some of the finest-level gap versus
the retained single-level formal benchmark is likely coming from:

- multi-task optimization tradeoffs
- shared-backbone capacity sharing
- the fact that finest-level prediction is no longer the only optimization
  target

So disabling hierarchy enforcement might improve finest-level metrics, but it is
unlikely by itself to guarantee full recovery to single-level formal baseline
quality.

## What would be lost by turning it off

The current round achieved:

- `mean_path_consistency_rate = 1.0`
- `min_path_consistency_rate = 1.0`

Disabling hierarchy enforcement would likely break this clean property. The
manuscript would then lose the strongest current evidence for guaranteed
hierarchy-consistent outputs.

## Recommendation

The most efficient next step is a predict-only ablation using the already
trained sixth-round models:

- keep training fixed
- keep `knn_correction="off"`
- compare:
  - `enforce_hierarchy=True`
  - `enforce_hierarchy=False`

This would directly answer:

1. how much finest-level performance is currently being traded for consistency
2. which datasets are most sensitive to the enforcement step
3. whether that tradeoff is acceptable for manuscript positioning

## Current working judgement

`hierarchy-aware` prediction is not obviously a uniformly beneficial choice if
the target objective is only finest-level headline performance. It does appear
to be highly beneficial if the target objective includes strict hierarchical
consistency. The right next step is therefore not a retraining round, but a
predict-only `hierarchy on/off` ablation.
