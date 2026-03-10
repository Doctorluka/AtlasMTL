# Phase 6A Execution Record

Date: `2026-03-10`

This round reuses the completed Phase 2 `lv4strong_plus_class_weight` seed models and evaluates
parent-conditioned rerankers for `top2`, `top4`, and `top6` hotspot parent sets.

## Hotspot Sets

```json
{
  "reranker_top2": [
    "CD4+ T",
    "SMC de-differentiated"
  ],
  "reranker_top4": [
    "CD4+ T",
    "SMC de-differentiated",
    "Mph alveolar",
    "EC vascular"
  ],
  "reranker_top6": [
    "CD4+ T",
    "SMC de-differentiated",
    "Mph alveolar",
    "EC vascular",
    "Fibro adventitial",
    "CD8+ T"
  ]
}
```

## Winner

```json
{
  "anno_lv4_balanced_accuracy_mean": 0.6178856850162057,
  "anno_lv4_balanced_accuracy_std": 0.0047726746494982765,
  "anno_lv4_coverage_mean": 0.89944,
  "anno_lv4_coverage_std": 0.008893143426258262,
  "anno_lv4_macro_f1_mean": 0.5873882447117136,
  "anno_lv4_macro_f1_std": 0.0023682437770089467,
  "delta_vs_baseline_anno_lv4_macro_f1_mean": 0.00021171674753195369,
  "delta_vs_baseline_full_path_accuracy_mean": 0.028300000000000002,
  "delta_vs_baseline_parent_correct_child_wrong_rate_mean": -0.028460000000000006,
  "full_path_accuracy_mean": 0.46665999999999996,
  "full_path_accuracy_std": 0.004926256184974548,
  "parent_correct_child_wrong_rate_mean": 0.09502,
  "parent_correct_child_wrong_rate_std": 0.0035152524802636864,
  "passes_gate": true,
  "path_break_rate_mean": 0.0,
  "path_break_rate_std": 0.0,
  "point": "predict_100000_10000",
  "variant_name": "reranker_top6"
}
```