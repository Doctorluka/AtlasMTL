# Fourth-Round Implementation Record (`2026-03-09`)

## Scope

This record tracks the implementation work for the fourth-round low-cost
atlasmtl optimization.

Primary implementation targets:

- add explicit optimizer selection for atlasmtl training
- add explicit weight decay control
- add optional `ReduceLROnPlateau` scheduling
- expose the new train fields through the benchmark runner
- add focused tests
- prepare round-specific execution assets

## Git checkpoint before code changes

- branch: `main`
- rollback anchor: `d351fe4`
- remote sync check before code changes: `origin/main` aligned with `HEAD`

## Execution notes

### Step 1: implementation planning

Status:

- completed

Actions:

- converted the round strategy into
  `plan/2026-03-09_fourth_round_atlasmtl_optimization_execution_plan.md`
- linked the execution plan from the dossier README

### Step 2: code implementation

Status:

- completed

Completed code touchpoints:

- `atlasmtl/core/train.py`
- `benchmark/pipelines/run_benchmark.py`
- `tests/unit/`
- `tests/integration/`

Implementation details:

- added explicit train-time optimizer controls:
  - `optimizer_name`
  - `weight_decay`
  - `scheduler_name`
  - `scheduler_factor`
  - `scheduler_patience`
  - `scheduler_min_lr`
  - `scheduler_monitor`
- preserved baseline behavior when no new train fields are provided
- added validation for unsupported optimizer/scheduler values
- required validation split when `ReduceLROnPlateau` is enabled
- recorded optimizer and scheduler metadata in `train_config`
- extended benchmark runner manifest validation and config passthrough for the
  new train keys

### Step 3: focused validation

Status:

- completed

Validation commands run:

```bash
python -m compileall atlasmtl benchmark tests
NUMBA_CACHE_DIR=/tmp/numba_cache /home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python -m pytest tests/unit/test_training_optimizer_config.py tests/integration/test_predict_integration.py -q
NUMBA_CACHE_DIR=/tmp/numba_cache /home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python -m pytest tests/integration/test_benchmark_runner.py -q -k optimizer_controls
```

Validation outcome:

- compile step passed
- `tests/unit/test_training_optimizer_config.py`: passed
- `tests/integration/test_predict_integration.py`: passed
- `tests/integration/test_benchmark_runner.py -k optimizer_controls`: passed

### Step 4: round execution assets

Status:

- completed for Stage A preparation assets

Created scripts:

- `scripts/freeze_baseline_table.py`
- `scripts/generate_low_cost_optimization_manifests.py`
- `scripts/collect_stage_results.py`
- `scripts/run_stage_a_cpu_core.sh`
- `scripts/run_stage_a_gpu.sh`
- `scripts/run_stage_b_cpu_core.sh`
- `scripts/run_stage_b_gpu.sh`

Smoke checks run:

```bash
/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/scripts/freeze_baseline_table.py
/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/scripts/generate_low_cost_optimization_manifests.py --stage stage_a
/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/scripts/collect_stage_results.py --stage stage_a
bash -n documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/scripts/run_stage_a_cpu_core.sh documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/scripts/run_stage_a_gpu.sh documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/scripts/run_stage_b_cpu_core.sh documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/scripts/run_stage_b_gpu.sh
```

Smoke check outcome:

- `stage_a_baseline_anchor.csv` generated with 8 rows
- Stage A manifest generation produced 32 manifests
- Stage A result collector ran successfully with `0` rows before benchmark execution
- all four shell scripts passed `bash -n`

## Validation checklist

- baseline train defaults unchanged when no new fields are provided
- invalid optimizer and scheduler names rejected cleanly
- scheduler requires validation split
- benchmark manifests can pass new train fields through to atlasmtl
- relevant unit and integration tests pass

## Runtime execution note

No Stage A or Stage B benchmark execution has started yet in this record.
