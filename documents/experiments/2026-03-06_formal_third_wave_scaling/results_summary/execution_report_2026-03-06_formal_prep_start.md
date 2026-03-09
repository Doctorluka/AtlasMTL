# Execution report: formal third-wave preparation start

Date: `2026-03-06`

## Scope

This report records the start of phase 1 for the formal third-wave scaling
round:

- data audit
- group-aware split generation
- preprocessing and prepared subset materialization

## Locked entry points

Plan:

- `plan/2026-03-06_formal_third_wave_scaling_plan.md`

Protocol:

- `documents/protocols/formal_third_wave_scaling_protocol.md`
- `documents/protocols/experiment_protocol.md`
- `documents/protocols/third_wave_fairness_protocol.md`

Round-level dossier:

- `documents/experiments/2026-03-06_formal_third_wave_scaling/`

## Active preparation script

Script:

- `documents/experiments/2026-03-06_formal_third_wave_scaling/scripts/prepare_formal_third_wave_scaling_inputs.py`

Dataset config:

- `documents/experiments/2026-03-06_formal_third_wave_scaling/configs/datasets.yaml`

Batch launcher:

- `documents/experiments/2026-03-06_formal_third_wave_scaling/scripts/run_formal_third_wave_prep_all.sh`

## Output root

Main tmp root:

- `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/`

Smoke tmp roots used during initial verification:

- `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave_smoke/`
- `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave_debug/`

## Current state

The phase-1 plan and protocol are recorded.

The new preparation script now implements:

- dataset audit
- ceiling search
- one formal split plan per dataset
- standalone `build_eval_fixed_10k`
- independent predict-scaling pool
- nested build tiers
- nested predict tiers
- preprocessing metadata and resource-summary outputs

The first live preparation checks were started with:

- `HLCA_Core` smoke run
- `Vento` smoke run
- `Vento` minimal-grid debug run

The next checkpoint is to confirm the first completed dataset outputs under the
tmp root and then launch the full dataset-preparation batch.

## Correction note

`Vento` required one post-run correction during the preparation phase.

Initial issue:

- the preparation script treated dataset-level `predict_tail_optional: []` as a
  falsy value and silently fell back to the global default optional tail
  `50k`
- this incorrectly forced `Vento` to request `heldout_total=60000`
- the side effect was an artificially collapsed feasible build ceiling
  (`build=10000`)

Fix:

- update list-override resolution so an explicit dataset-level empty list
  remains an empty list and does not fall back to defaults
- rerun `Vento` preparation after removing the incorrect tmp output

Corrected `Vento` result:

- build grid feasible: `10k / 20k / 30k / 50k`
- fixed build for predict scaling: `50k`
- predict grid feasible: `1k / 3k / 5k / 8k / 10k`
- optional predict tail: disabled

Corrected output root:

- `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/Vento/prepared/formal_split_v1/`
