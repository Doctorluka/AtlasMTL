# Phase 7A Auto Reranker Pipeline

- model seed: `2026`
- base model: `lv4strong_plus_class_weight`
- refinement mode: `auto_parent_conditioned_reranker`
- selection source: `Phase 3 baseline`
- selection score: `parent_correct_child_wrong_rate * n_cells`
- selected parents: `CD4+ T, SMC de-differentiated, Mph alveolar, EC vascular, Fibro adventitial, CD8+ T`
- guardrail passed: `True`

## Outputs

- `artifacts/phase7a_auto_reranker_pipeline/hotspot_ranking.json`
- `artifacts/phase7a_auto_reranker_pipeline/refinement_plan.json`
- `artifacts/phase7a_auto_reranker_pipeline/parent_conditioned_reranker_top6.pkl`
- `artifacts/phase7a_auto_reranker_pipeline/per_parent_reranker_summary.csv`
- `artifacts/phase7a_auto_reranker_pipeline/guardrail_decision.json`
- `results_summary/phase7a_auto_reranker_pipeline/phase7a_auto_reranker_pipeline.csv`
- `results_summary/phase7a_auto_reranker_pipeline/phase7a_auto_reranker_parent_child_breakdown.csv`