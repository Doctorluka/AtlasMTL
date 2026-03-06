# Scanvi parameter-lock benchmark dossier (`2026-03-06`)

This dossier stores the pre-formal expanded benchmark used to lock default `scanvi` parameters.

## Positioning

- scope: `experiment` (pre-formal)
- objective: parameter confirmation for methods-section evidence
- method: `scanvi` only
- execution mode: GPU only (`scanvi` is not evaluated on CPU in this round)

## Directory layout

- `configs/`
  - dataset registry and parameter grid
- `scripts/`
  - input materialization, sweep execution, aggregation, table rendering
- `results_summary/`
  - execution reports, decision notes, exported tables

## Reports

- primary experiment report:
  - `results_summary/experiment_report_2026-03-06_scanvi_param_lock.md`
- execution report:
  - `results_summary/execution_report_2026-03-06_scanvi_param_lock.md`
- lock decision note:
  - `results_summary/param_lock_decision_2026-03-06.md`
- error-fix note:
  - `results_summary/error_fix_note_2026-03-06_scanvi_param_lock.md`

The experiment report is the main citation entry for methods writing; the
execution/error notes are the reproducibility and troubleshooting records.

## Runtime roots

- prepared inputs:
  - `/tmp/atlasmtl_benchmarks/2026-03-06/reference_heldout/<dataset>/prepared/param_lock_train10k/`
- stage A outputs:
  - `/tmp/atlasmtl_benchmarks/2026-03-06/scanvi_param_lock/stage_a/`
- stage B outputs:
  - `/tmp/atlasmtl_benchmarks/2026-03-06/scanvi_param_lock/stage_b/`

## Execution note

- sandbox runs are valid for pipeline smoke only.
- for real `scanvi` GPU parameter confirmation, execute from a non-sandbox
  shell because sandboxed CUDA visibility is not reliable.
- CPU-mode `scanvi` runs are excluded from this benchmark by design.

## Quick start

```bash
python documents/experiments/2026-03-06_scanvi_param_lock_benchmark/scripts/prepare_scanvi_param_lock_inputs.py
python documents/experiments/2026-03-06_scanvi_param_lock_benchmark/scripts/run_scanvi_param_sweep.py --stage stage_a
python documents/experiments/2026-03-06_scanvi_param_lock_benchmark/scripts/aggregate_scanvi_param_sweep.py
python documents/experiments/2026-03-06_scanvi_param_lock_benchmark/scripts/run_scanvi_param_sweep.py --stage stage_b
python documents/experiments/2026-03-06_scanvi_param_lock_benchmark/scripts/aggregate_scanvi_param_sweep.py
python documents/experiments/2026-03-06_scanvi_param_lock_benchmark/scripts/render_scanvi_param_tables.py
```
