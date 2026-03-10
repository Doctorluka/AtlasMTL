# Phase 2 Execution Record

Date: `2026-03-10`

This record captures the PH-Map study-split follow-up round that implements the
expert-recommended finest-level imbalance handling screen and the `N=5` seed
stability check.

## Scope

- dataset: `PHMap_Lung_Full_v43_light`
- split protocol: `study`
- prepared root:
  - `/tmp/atlasmtl_benchmarks/2026-03-09/multilevel_annotation_study_split/PHMap_Lung_Full_v43_light/prepared/formal_split_v1/`
- track: `gpu`
- fixed training contract:
  - `task_weights = [0.2, 0.7, 1.5, 3.0]`
  - `optimizer_name=adamw`
  - `weight_decay=5e-5`
  - `scheduler_name=null`
  - `num_epochs=50`
  - `input_transform=binary`
  - `knn_correction=off`
  - `enforce_hierarchy=true`

## Implemented code changes

- added optional `class_weighting` support to `atlasmtl.build_model()`
- added optional `class_balanced_sampling` support to `atlasmtl.build_model()`
- wired both controls through `benchmark/pipelines/run_benchmark.py`
- added focused unit coverage for manifest/API validation and train-config
  recording

## Phase 2A: single-seed imbalance screen

Screened configurations:

- `lv4strong_baseline`
- `lv4strong_plus_class_weight`
- `lv4strong_plus_class_balanced_sampling`
- `lv4strong_plus_both`

Representative findings on `predict_100000_10000`:

- `baseline`: `macro_f1=0.565757`, `full_path_accuracy=0.4720`,
  `coverage=0.7920`
- `+ class_weight`: `macro_f1=0.587630`, `full_path_accuracy=0.4437`,
  `coverage=0.7714`
- `+ class_balanced_sampling`: `macro_f1=0.575706`,
  `full_path_accuracy=0.4476`, `coverage=0.8758`
- `+ both`: `macro_f1=0.521184`, `full_path_accuracy=0.3844`,
  `coverage=0.8091`

Selection result:

- `lv4strong_plus_class_weight` was selected as the best seed-stability
  candidate because it gave the strongest finest-level gain on the primary
  `predict_100000_10000` point.

## Phase 2B: seed stability (`N=5`)

Compared configurations:

- `lv4strong_baseline`
- `lv4strong_plus_class_weight`

Seeds:

- `2026`
- `17`
- `23`
- `47`
- `101`

Predict-point aggregate:

- `macro_f1 mean/std`: `0.56845 ± 0.01222` -> `0.58718 ± 0.00570`
- `balanced_accuracy mean/std`: `0.56552 ± 0.01090` -> `0.58651 ± 0.00873`
- `coverage mean/std`: `0.76550 ± 0.01359` -> `0.78234 ± 0.02204`
- `unknown_rate mean/std`: `0.23450 ± 0.01359` -> `0.21766 ± 0.02204`
- `full_path_accuracy mean/std`: `0.46904 ± 0.01141` -> `0.43836 ± 0.01042`

## Decision implication

- `finest-level per-class weighting` is now the best-supported next-step
  improvement for PH-Map under `study` split.
- The gain is not only reproducible across the stricter split, but also stable
  across seeds on `anno_lv4 macro_f1` and `balanced_accuracy`.
- The main remaining tradeoff is a regression in `full_path_accuracy`; this
  should be stated explicitly if the weighting is promoted as a finest-level
  default candidate.
