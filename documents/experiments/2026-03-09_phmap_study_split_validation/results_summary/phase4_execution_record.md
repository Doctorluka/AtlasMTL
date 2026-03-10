# Phase 4 Execution Record

Date: `2026-03-10`

This round applies hotspot-local child refinement on top of `lv4strong_plus_class_weight`.

- hotspot parents: `CD4+ T, CD8+ T, Mph alveolar, Pericyte`
- hotspot thresholds: `max_prob<0.55`, `margin<0.15`
- hotspot temperature: `T=1.25`

Compared variants:

- `baseline`
- `hotspot_thresholding`
- `hotspot_temperature_scaling`
- `hotspot_thresholding_plus_temperature`

Main result:

- no refinement variant passed the Phase 4 guardrail
- all non-baseline variants reduced `anno_lv4 macro_f1`
- none improved `full_path_accuracy`
- none reduced `parent_correct_child_wrong_rate`

Representative `predict_100000_10000 + hierarchy_on` deltas:

- `hotspot_thresholding`
  - `macro_f1`: `-0.0171`
  - `full_path_accuracy`: `-0.0143`
  - `parent_correct_child_wrong_rate`: `+0.0143`
- `hotspot_temperature_scaling`
  - `macro_f1`: `-0.0102`
  - `full_path_accuracy`: `-0.0064`
  - `parent_correct_child_wrong_rate`: `+0.0064`
- `hotspot_thresholding_plus_temperature`
  - `macro_f1`: `-0.0320`
  - `full_path_accuracy`: `-0.0250`
  - `parent_correct_child_wrong_rate`: `+0.0251`

Decision implication:

- hotspot-local thresholding and simple shared temperature scaling are not
  sufficient fixes for the current PH-Map tradeoff
- the negative result strengthens the case that the remaining issue is more
  structural than purely post-hoc
- the next justified direction, if continued, is a more explicit
  parent-conditioned child refinement or a lightweight hierarchy-aware training
  change
