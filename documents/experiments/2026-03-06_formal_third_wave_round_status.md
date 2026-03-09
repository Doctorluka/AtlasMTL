# Formal third-wave round status

Date: `2026-03-06`

Current round: formal third-wave scaling.

## Locked roster

Main panel:

- `HLCA_Core`
- `PHMap_Lung_Full_v43_light`
- `mTCA`
- `DISCO_hPBMCs`

Supplementary:

- `Vento`

Excluded:

- `cd4`
- `cd8`

## Current phase

Phase 1 completed:

- data audit
- split generation
- preprocessing and prepared subset materialization

Phase 2 started with `HLCA_Core` sanity execution:

- GPU sanity completed for `build100k -> eval10k` and `predict100k -> 10k`
- CPU sanity was stopped after a long-running `seurat_anchor_transfer` step at
  `build100k -> eval10k`
- long-runtime stop rule is now locked: single-method runs over `60 minutes`
  should be manually stopped and recorded
- formal `HLCA_Core` main batch launchers are prepared with three tracks:
  `gpu`, `cpu_core`, and isolated `cpu_seurat`
- current launch state:
  - `cpu_core`: completed
  - `cpu_seurat`: partial, `build_10000_eval10k` completed
  - `gpu`: must be launched from a direct non-sandbox shell when CUDA is not
    available in the sandboxed execution context

## Preparation status checklist

- [x] `HLCA_Core`
- [x] `PHMap_Lung_Full_v43_light`
- [x] `mTCA`
- [x] `DISCO_hPBMCs`
- [x] `Vento` supplementary

## Key rules carried into this round

- build grid: `10k / 20k / 30k / 50k / 100k / 150k / 200k / 300k`
- build scaling fixed query: standalone `10k`
- predict scaling fixed build: reuse `100k` artifact
- predict grid: `1k / 3k / 5k / 8k / 10k / 15k / 20k`
- optional predict tail: `50k`
- build-scaling `10k` and predict-scaling `10k` are different subsets
- `Vento` is supplementary reduced-ceiling

## Phase-1 outputs

Prepared assets:

- `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/<dataset>/prepared/formal_split_v1/`

Round-level report:

- `documents/experiments/2026-03-06_formal_third_wave_scaling/results_summary/phase1_preparation_report_2026-03-06.md`

Manifest index:

- `documents/experiments/2026-03-06_formal_third_wave_scaling/manifests/reference_heldout/manifest_index.json`

HLCA sanity execution record:

- `documents/experiments/2026-03-06_formal_third_wave_scaling/results_summary/execution_report_2026-03-06_formal_hlca_sanity_start.md`

## Queue restart status

On `2026-03-08`, the remaining main-panel queues were confirmed to have no
active worker processes after an interruption. Formal execution therefore
continues from persisted checkpoint state instead of assuming the queues are
still live.

Restart rule:

- restart `cpu_core`, `gpu`, and `cpu_seurat` queue launchers
- skip only fully successful points
- preserve partial outputs for audit, but rerun incomplete points as needed
