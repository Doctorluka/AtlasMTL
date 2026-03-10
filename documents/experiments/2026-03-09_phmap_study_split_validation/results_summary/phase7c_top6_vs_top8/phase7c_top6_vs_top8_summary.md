# Phase 7C top6 vs top8 seed stability

- base model: `lv4strong_plus_class_weight`
- refinement method: `auto parent-conditioned reranker`
- seeds: `101, 17, 2026, 23, 47`
- primary point: `predict_100000_10000 + hierarchy_on`

## Decision

- `Promote top8 to PH-Map default hotspot rule`

## Primary comparison

- `reranker_top6`: macro_f1 `0.587355 ± 0.002368`, full_path `0.46664 ± 0.00493`, parent_correct_child_wrong `0.09504 ± 0.00353`
- `reranker_top8`: macro_f1 `0.588557 ± 0.002093`, full_path `0.47216 ± 0.00453`, parent_correct_child_wrong `0.08926 ± 0.00310`