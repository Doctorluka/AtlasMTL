# Phase 3 Execution Record

Date: `2026-03-10`

This record captures the PH-Map tradeoff-attribution follow-up round that
treats `lv4strong_plus_class_weight` as the current PH-Map baseline candidate.

## Scope

- dataset: `PHMap_Lung_Full_v43_light`
- split protocol: `study`
- prepared root:
  - `/tmp/atlasmtl_benchmarks/2026-03-09/multilevel_annotation_study_split/PHMap_Lung_Full_v43_light/prepared/formal_split_v1/`
- compared configs:
  - `lv4strong_baseline`
  - `lv4strong_plus_class_weight`
- fixed training contract:
  - `task_weights = [0.2, 0.7, 1.5, 3.0]`
  - `optimizer_name=adamw`
  - `weight_decay=5e-5`
  - `scheduler_name=null`
  - `num_epochs=50`
  - `input_transform=binary`
  - `knn_correction=off`
  - seed `2026`

## Executed matrix

- train runs: `2`
- predict evaluations: `8`
- points:
  - `build_100000_eval10k`
  - `predict_100000_10000`
- hierarchy settings:
  - `enforce_hierarchy=true`
  - `enforce_hierarchy=false`

## Main findings

- `lv4strong_plus_class_weight` remains better on finest-level metrics under
  both `hierarchy_on` and `hierarchy_off`.
- On `predict_100000_10000 + hierarchy_on`, finest-level `macro_f1` improved
  from `0.560001` to `0.591375`, while `full_path_accuracy` decreased from
  `0.4562` to `0.4443`.
- On `predict_100000_10000 + hierarchy_off`, `path_break_rate` on the
  `anno_lv3 -> anno_lv4` edge increased from `0.0400` to `0.0557`, and
  `path_consistency_rate` decreased from `0.9505` to `0.9372`.
- The largest change on the `anno_lv3 -> anno_lv4` edge is the increase in
  `parent_correct_child_wrong_rate`:
  - `predict_100000_10000 + hierarchy_on`: `0.1003 -> 0.1175`
  - `predict_100000_10000 + hierarchy_off`: `0.1014 -> 0.1188`
- This means the full-path regression is driven primarily by more child-level
  errors under otherwise correct parents, not only by explicit hierarchy
  inconsistency.

## Localized hotspots

The strongest `anno_lv3` subtree hotspots under
`lv4strong_plus_class_weight + predict_100000_10000 + hierarchy_off` are:

- `CD8+ T`
- `Mph alveolar`
- `Pericyte`
- `CD4+ T`

These subtrees show elevated `path_break_rate` and/or
`parent_correct_child_wrong_rate`, so any next-step hierarchy-aware refinement
should be justified against these concentrated failure modes rather than a
global hierarchy mismatch story.

## Decision implication

- Primary tradeoff category: `child discrimination tradeoff`
- Secondary tradeoff category: `hierarchy inconsistency tradeoff`
- `hierarchy_on` is not sufficient to remove the `full_path_accuracy` gap
  because the dominant error mode is wrong-child-under-correct-parent.
- If a next code change is considered, the strongest justification would be a
  lightweight hierarchy-aware loss or targeted subtree-specific handling, not a
  return to sampling or optimizer search.
