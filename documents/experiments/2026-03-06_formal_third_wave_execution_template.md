# Formal third-wave execution template

Date: `2026-03-06`

Use this template when recording the formal third-wave preparation phase and
the later benchmark phase.

## What to record first

- dataset name
- panel type: `main` or `supplementary`
- source `.h5ad`
- prep manifest path
- split key
- target label
- build grid requested
- build grid feasible
- predict grid requested
- predict grid feasible
- whether optional `50k` predict tail is available

## Required preparation outputs

Under `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/<dataset>/prepared/formal_split_v1/`:

- `split_plan.json`
- `split_summary.json`
- `preprocessing_summary.json`
- `preparation_resource_summary.json`
- `dataset_ceiling_summary.json`

Prepared assets must also make these distinctions explicit:

- `build_eval_fixed_10k`
- `heldout_predict_10k`
- fixed-build artifact reused for predict scaling

## Required report files

Round-level:

- `documents/experiments/2026-03-06_formal_third_wave_round_status.md`
- `documents/experiments/2026-03-06_formal_third_wave_round_report.md`

Round-level preparation dossier:

- `documents/experiments/2026-03-06_formal_third_wave_scaling/results_summary/`

## Mandatory protocol reminders

- `scanvi` is GPU-only
- `atlasmtl_cpu` and `atlasmtl_gpu` are separate rows
- `celltypist` backend path must be labeled
- the build-scaling `10k` query cannot be reused as predict-scaling `10k`
- all methods for one dataset must consume the same prepared subset files
