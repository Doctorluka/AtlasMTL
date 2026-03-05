# CellTypist comparator gap note

Date: `2026-03-04`

## Problem

The current benchmark-facing `celltypist` training path is not equivalent to
the earlier ProjectSVR-style CellTypist training workflow used in historical
PH-Map benchmarking scripts.

## Historical reference workflow

Reference file:

- `/home/data/fhz/project/phmap_package/scripts/04_benchmark_build_models.py`

Historical training characteristics:

- calls `celltypist.train(...)` directly
- trains separate models for multiple label levels
- uses `feature_selection=True`
- uses `balance_cell_type=True`
- uses explicit `top_genes`
- uses parallel workers via `n_jobs`

This is the workflow whose model-build time was visualized in:

- `/home/data/fhz/project/phmap_package/scripts/06_benchamrk_vis.ipynb`

## Current benchmark implementation gap

Current wrapper path:

- `documents/experiments/2026-03-01_real_mapping_benchmark/scripts/train_celltypist_model.py`

Current behavior:

- does **not** call `celltypist.train(...)`
- trains a plain `sklearn.linear_model.LogisticRegression`
- wraps that estimator as a `celltypist.models.Model`
- trains only one target label per run
- omits historical CellTypist feature-selection and balancing steps

## Consequence

The currently reported `celltypist` build time in recent benchmark reports is
**not directly comparable** to historical PH-Map CellTypist model-build timing.

In particular:

- very small train times for recent benchmark runs reflect the lightweight
  wrapped-logistic implementation
- they should not be interpreted as the cost of the historical CellTypist
  training pipeline

## Required reporting rule

Until the formal trainer path becomes the default:

- benchmark reports must treat the current implementation as a
  **simplified CellTypist comparator**
- any timing comparison against historical PH-Map CellTypist training must be
  labeled as not directly comparable

## Resolution plan

Introduce a formal CellTypist trainer option that:

- calls `celltypist.train(...)`
- exposes key historical knobs such as:
  - `feature_selection`
  - `balance_cell_type`
  - `top_genes`
  - `n_jobs`
  - `batch_size`
  - `max_iter`

The simplified trainer may remain temporarily for engineering smoke runs, but
formal benchmark interpretation should prefer the explicit formal trainer path.

## Implemented interface

The scale-out and smoke wrappers now accept:

- `method_configs.celltypist.trainer_backend`
  - `wrapped_logreg` (current lightweight default)
  - `formal` (calls `celltypist.train(...)`)
- `method_configs.celltypist.trainer_config`
  - `max_iter`
  - `n_jobs`
  - `feature_selection`
  - `balance_cell_type`
  - `batch_size`
  - `top_genes`
  - `use_gpu`
  - `with_mean`
  - `min_cells_per_class`

The trainer writes a JSON summary alongside the generated comparator model so
benchmark records can distinguish lightweight vs formal CellTypist builds.

## Formal smoke-test status

Status as of `2026-03-04`:

- a first small-scale formal smoke test was attempted on a `DISCO`-derived
  `1k reference / 1k query` setup
- input preparation completed successfully
- the formal training path did **not** complete a valid end-to-end run yet

Observed blocker:

- direct use of `celltypist.train(...)` exposed an environment compatibility
  issue with the current local stack:
  - `celltypist==1.7.1`
  - `scikit-learn==1.8.0`
- the failing path internally attempted to construct
  `LogisticRegression(..., multi_class='ovr', ...)`
- in the current local scikit-learn version, this argument is not accepted in
  the same way, causing the formal smoke test to fail before a valid comparator
  model was produced

Current repository state:

- a compatibility wrapper and a `trainer_backend=formal` interface have already
  been added in code
- however, the formal CellTypist path is still considered **not yet validated**
  until a full small-scale smoke test produces:
  - a valid `.pkl` model
  - a `training_summary.json`
  - a successful downstream prediction run

Interim rule:

- until that validation is complete, active benchmark reports must continue to
  label the current benchmark-facing `celltypist` results as
  `wrapped_logreg`-based and not as fully validated formal CellTypist runs
