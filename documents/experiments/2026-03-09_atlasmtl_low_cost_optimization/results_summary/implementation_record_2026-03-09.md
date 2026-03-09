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

## Stage A CPU execution (`2026-03-09`)

Status:

- completed for `cpu_core`

Execution command:

```bash
bash documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/scripts/run_stage_a_cpu_core.sh
```

Execution mode note:

- runs were executed in the restricted Codex environment
- every CPU run reported `joblib_serial_fallback`
- therefore runtime numbers should be treated as provisional fairness data, not
  final paper-grade resource evidence

Observed CPU Stage A screening summary:

- `PHMap_Lung_Full_v43_light / build_100000_eval10k`
  - baseline `macro_f1=0.642994`
  - `wd=1e-5` `macro_f1=0.625151`
  - `wd=5e-5` `macro_f1=0.647272`
  - `wd=1e-4` `macro_f1=0.640712`
- `PHMap_Lung_Full_v43_light / predict_100000_10000`
  - baseline `macro_f1=0.651921`
  - `wd=1e-5` `macro_f1=0.648428`
  - `wd=5e-5` `macro_f1=0.651283`
  - `wd=1e-4` `macro_f1=0.635325`
- `mTCA / build_100000_eval10k`
  - baseline `macro_f1=0.845717`
  - `wd=1e-5` `macro_f1=0.865667`
  - `wd=5e-5` `macro_f1=0.869528`
  - `wd=1e-4` `macro_f1=0.837774`
- `mTCA / predict_100000_10000`
  - baseline `macro_f1=0.846167`
  - `wd=1e-5` `macro_f1=0.869166`
  - `wd=5e-5` `macro_f1=0.859251`
  - `wd=1e-4` `macro_f1=0.852248`

Provisional CPU-only interpretation:

- `wd=1e-5` is not a robust candidate because it underperforms on both PHMap points
- `wd=1e-4` is not a robust candidate because it harms PHMap predict and mTCA build
- `wd=5e-5` is the strongest compromise on CPU:
  - slightly better than baseline on difficult `PHMap build`
  - near-neutral on `PHMap predict`
  - clearly better than baseline on both `mTCA` points

Next step:

- Stage A GPU was subsequently executed outside the sandbox
- final Stage A candidate is `AdamW + wd=5e-5`
- scheduler check completed and was rejected

Stage A lock after review:

- proceed to Stage B with `AdamW + wd=5e-5` as the only candidate default
- stop the scheduler branch completely
- do not expand the `weight_decay` grid further
- Stage B should answer whether the candidate remains broadly default-acceptable,
  not whether it is strictly better on every single representative point
- the isolated `mTCA gpu predict` regression should remain a watchpoint in Stage
  B, but not a standalone reason to reject the candidate before confirmation
