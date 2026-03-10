# Parent-Conditioned Reranker Chapter Design Summary

## Chapter positioning

This chapter is framed as an optional structured refinement module for difficult deep-hierarchy cases rather than as a second core AtlasMTL model.

## Core claim under evaluation

For difficult deep-hierarchy reference mapping, error-driven parent-conditioned hotspot reranking can improve finest-level annotation and full-path hierarchy recovery beyond the best base multi-level AtlasMTL configuration.

## Experimental design

### PH-Map

- Rebuild the evaluation around a stricter `study`-isolated split from raw PH-Map reference data.
- Confirm that finest-level task emphasis and finest-head class weighting are the highest-ROI base-model improvements.
- Diagnose the remaining tradeoff through parent-child error decomposition.
- Reject naive local fixes such as hotspot thresholding and shared temperature scaling.
- Promote the auto parent-conditioned reranker to the operational path and select the default hotspot rule through multi-seed `top6` vs `top8` confirmation.

### HLCA

- Rebuild HLCA from raw `hlca_clean.h5ad` using a `study`-grouped split instead of reusing a legacy benchmark subset.
- Because HLCA has five annotation levels, run a dataset-specific 5D weighting confirmation rather than inheriting PH-Map weights.
- Use the winning HLCA base configuration to test whether the AutoHotspot reranker mechanism transfers to a second deep-hierarchy dataset.
- Narrow the reranker-rule follow-up to `top4` vs `top6` rather than reopening a larger hotspot-width search.
- Add a small policy validation step that asks whether a dataset should leave `uniform` task weights at all before entering a weighting candidate test.

## Figure plan

- Main Figure Panel A: final variant comparison.
- Main Figure Panel B: error-mode comparison.
- Main Figure Panel C: method and artifact schematic support.
- Main Figure Panel D: stability/default-rule support.
- Supplementary S1-S5: ablation ladder, stability/by-group, hotspot-rule comparison, train-time internalization branch, weight-activation policy support.
