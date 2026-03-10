# HLCA AutoHotspot Reranker Validation

- best base config: `uniform`
- target edge: `ann_level_4 -> ann_level_5`
- selection rule: `parent_correct_child_wrong_rate * n_cells`, `topk=6`, `min_cells_per_parent=200`
- selected hotspot parents: Alveolar macrophages, Interstitial macrophages, Goblet, Club, Multiciliated, AT2

## Target point

- baseline: macro_f1 `0.688732`, full_path `0.8239`, parent_correct_child_wrong `0.0334`
- auto reranker: macro_f1 `0.693015`, full_path `0.8200`, parent_correct_child_wrong `0.0371`
- guardrail pass: `False`

## Outputs

- `hlca_main_comparison.csv`
- `hlca_error_decomposition.csv`
- `hlca_subtree_breakdown.csv`
- `hlca_by_study.csv`
- `hlca_hotspot_ranking.json`
- `hlca_refinement_plan.json`
- `hlca_guardrail_decision.json`
