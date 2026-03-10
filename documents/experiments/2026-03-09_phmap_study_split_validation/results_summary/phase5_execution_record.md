# Phase 5 Execution Record

Date: `2026-03-10`

This round applies parent-conditioned hotspot child reranking on top of `lv4strong_plus_class_weight`.

- hotspot parents: `CD8+ T, Mph alveolar, Pericyte, CD4+ T`
- fitted rerankers: `CD4+ T, CD8+ T, Mph alveolar, Pericyte`

Winner:

{
  "anno_lv4_macro_f1": 0.6021965166599931,
  "delta_vs_baseline_anno_lv4_macro_f1": 0.010821751252037437,
  "delta_vs_baseline_full_path_accuracy": 0.01870000000000005,
  "full_path_accuracy": 0.463,
  "hierarchy_setting": "on",
  "point": "predict_100000_10000",
  "status": "selected",
  "winner_variant": "parent_conditioned_reranker"
}
