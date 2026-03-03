# First-wave split strategy (trial protocol)

Date: `2026-03-03`

This document defines the current **trial** split strategy for the first
flow-through benchmark wave. It is designed to make the pipeline executable
and auditable, not to serve as the final paper-grade split standard.

## Scope

This split protocol currently applies only to:

- `PHMap_Lung_Full_v43_light`
- `DISCO_hPBMCs`

Locked first-wave subset sizes:

- reference build subset: `5k`
- heldout prediction subset: `1k`

Locked first-wave split/report key:

- `split_key = sample`
- `domain_key = sample`

## Terminology

- `split_key`
  - group field used to prevent train/test leakage
- `domain_key`
  - grouped reporting field used for outputs such as `summary_by_domain.csv`
- `build pool`
  - group-isolated pool used to construct the reference subset
- `predict pool`
  - group-isolated pool used to construct the heldout prediction subset

## Strategy: A+

The first-wave algorithm is **A+**:

1. enumerate groups from `obs[split_key]`
2. generate multiple random candidate group splits
3. reject invalid candidates
4. score valid candidates
5. choose the best candidate deterministically
6. sample exact cell counts from the chosen pools

This is intentionally simpler than a final hierarchical or label-stratified
production split, but stronger than naive random cell-level splitting.

## Candidate generation

For each candidate:

1. shuffle the unique groups with a seeded RNG
2. assign groups greedily into:
   - build pool
   - predict pool
3. continue until both target capacities are feasible

Current defaults:

- `seed = 2026`
- `n_candidates = 128`

## Candidate rejection rules

Reject a candidate if any of the following happens:

- group leakage exists between build and predict pools
- build pool has fewer than `5k` cells
- predict pool has fewer than `1k` cells
- build pool collapses to fewer than `2` target labels
- predict pool collapses to fewer than `2` target labels

## Candidate scoring

Valid candidates are ranked by:

1. lower size overshoot beyond the target
2. better heldout label coverage
3. better heldout minimum label support
4. better build label coverage
5. better overall pool balance

If candidates tie, choose deterministically by sorted group identity.

## Post-selection subset materialization

After choosing the best candidate:

- sample exactly `5k` cells from the build pool
- sample exactly `1k` cells from the predict pool

Current first-wave rule:

- random sampling with fixed seed
- no mandatory label-stratified sampling in the first implementation
- perform post-sampling label checks and record warnings

## Validation rule

First-wave execution does **not** create a third external validation asset.

Instead:

- external split = build subset vs predict subset
- model-internal validation continues to use `train.val_fraction`

## Warning policy

The first wave is warning-first, not fail-first, except for structural errors.

Hard fail:

- subset target size cannot be reached
- fewer than `2` target labels remain in a sampled subset

Warning:

- build subset contains a target label with `<10` cells
- predict subset contains a target label with `<5` cells

Warnings must be written into the split summary.

## Required split outputs

Every materialized split should emit:

- `split_plan.json`
- `split_summary.json`

Required contents include:

- dataset id
- source dataset path
- split key
- domain key
- target label
- random seed
- candidate count
- chosen candidate id
- build groups
- predict groups
- build/predict pool sizes
- sampled build/predict sizes
- build/predict label distributions
- warnings
- aggregated rejection reasons

## First-wave limitations

- this protocol is not yet the final paper-grade split standard
- it optimizes for leakage control and reproducibility first
- it does not yet enforce full label-stratified sampling
- later rounds may add:
  - stricter label support thresholds
  - multi-level-aware split constraints
  - explicit shift scenarios
