# Phase 1 Execution Record

Date: `2026-03-09`

This record captures the PH-Map Phase 1 rerun under the independent
`study`-split validation dossier.

## Scope

- dataset: `PHMap_Lung_Full_v43_light`
- split protocol: `study`
- prepared root:
  - `/tmp/atlasmtl_benchmarks/2026-03-09/multilevel_annotation_study_split/PHMap_Lung_Full_v43_light/prepared/formal_split_v1/`
- track: `gpu`
- training contract:
  - `input_transform=binary`
  - `optimizer_name=adamw`
  - `weight_decay=5e-5`
  - `scheduler_name=null`
  - `num_epochs=50`
  - `knn_correction=off`

## Compared configurations

- `uniform_control = [1.0, 1.0, 1.0, 1.0]`
- `lv4strong_candidate = [0.2, 0.7, 1.5, 3.0]`

## Executed matrix

- train runs: `2`
- predict evaluations: `8`
- representative points:
  - `build_100000_eval10k`
  - `predict_100000_10000`
- hierarchy settings:
  - `enforce_hierarchy=true`
  - `enforce_hierarchy=false`

## Main findings

- `lv4strong_candidate` beat `uniform_control` in all four
  point-by-hierarchy cells.
- On `predict_100000_10000 + hierarchy_on`, finest-level `macro_f1` improved
  from `0.541423` to `0.552978`, `full_path_accuracy` improved from `0.4570`
  to `0.4639`, and `coverage` improved from `0.7574` to `0.7668`.
- On `build_100000_eval10k + hierarchy_on`, finest-level `macro_f1` improved
  from `0.549254` to `0.560272`, `full_path_accuracy` improved from `0.4548`
  to `0.4591`, and `coverage` improved from `0.7663` to `0.7764`.
- Turning hierarchy enforcement off increased coverage by about `+0.026` for
  `uniform_control` and about `+0.040` for `lv4strong_candidate`, while
  reducing mean path consistency from `1.0` to about `0.983` and `0.975`
  respectively.

## Decision implication

- The PH-Map weighting result survives the stricter `study`-isolated split.
- `lv4strong_candidate` remains the current leading PH-Map weighting candidate.
- The next validation step should focus on repeatability and fine-level
  imbalance handling rather than optimizer retuning.
