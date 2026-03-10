# PH-Map reranker_top6 operational module v1

- model seed: `2026`
- source model: `lv4strong_plus_class_weight`
- refinement method: `parent_conditioned_reranker`
- refinement edge: `anno_lv3 -> anno_lv4`
- hotspot parents: `CD4+ T, SMC de-differentiated, Mph alveolar, EC vascular, Fibro adventitial, CD8+ T`
- selection rule: `Phase 3 baseline`, `predict_100000_10000 + hierarchy_on`, `parent_correct_child_wrong_rate * n_cells`

## Outputs

- `hotspot_ranking.json`
- `parent_conditioned_reranker_top6.pkl`
- `parent_conditioned_reranker_top6.json`
- `per_parent_reranker_summary.csv`
- `before_after_comparison.csv`
- `before_after_parent_child_breakdown.csv`