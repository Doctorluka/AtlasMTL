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

## Stage B CPU execution (`2026-03-09`)

Status:

- completed for `cpu_core`

Execution command:

```bash
bash documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/scripts/run_stage_b_cpu_core.sh
```

Execution mode note:

- runs were executed in the restricted Codex environment
- every CPU run reported `joblib_serial_fallback`
- CPU runtime and throughput numbers remain provisional fairness evidence only

Observed Stage B CPU summary (`macro_f1`, candidate minus baseline):

- `HLCA_Core / build_100000_eval10k`: `-0.022878`
- `HLCA_Core / predict_100000_10000`: `-0.013927`
- `PHMap_Lung_Full_v43_light / build_100000_eval10k`: `-0.001764`
- `PHMap_Lung_Full_v43_light / predict_100000_10000`: `+0.000693`
- `mTCA / build_100000_eval10k`: `-0.005925`
- `mTCA / predict_100000_10000`: `+0.039689`
- `DISCO_hPBMCs / build_100000_eval10k`: `-0.028083`
- `DISCO_hPBMCs / predict_100000_10000`: `+0.008827`

CPU interpretation:

- CPU evidence alone is mixed and does not justify a default promotion
- the strongest positive CPU signal remains `mTCA predict`
- the strongest negative CPU signals are `HLCA` and `DISCO build`

## Stage B GPU execution (`2026-03-09`)

Status:

- completed for `gpu`

Execution command:

```bash
bash documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/scripts/run_stage_b_gpu.sh
```

Execution mode note:

- runs were executed outside the sandbox on `NVIDIA GeForce RTX 4090`
- GPU runs did not report runtime fairness degradation

Observed Stage B GPU summary (`macro_f1`, candidate minus baseline):

- `HLCA_Core / build_100000_eval10k`: `+0.014619`
- `HLCA_Core / predict_100000_10000`: `+0.005678`
- `PHMap_Lung_Full_v43_light / build_100000_eval10k`: `+0.016954`
- `PHMap_Lung_Full_v43_light / predict_100000_10000`: `-0.008676`
- `mTCA / build_100000_eval10k`: `+0.014338`
- `mTCA / predict_100000_10000`: `+0.021822`
- `DISCO_hPBMCs / build_100000_eval10k`: `+0.034498`
- `DISCO_hPBMCs / predict_100000_10000`: `+0.041895`

GPU interpretation:

- candidate improves `7/8` representative GPU points
- candidate preserves the previously watched `mTCA gpu predict` point and turns
  it into a positive result
- the only GPU regression is `PHMap predict`, and it is materially smaller than
  the gains seen on `PHMap build`, `mTCA`, and `DISCO`

## Stage B final interpretation

- CPU evidence is mixed but runtime-degraded
- GPU evidence is strong, non-degraded, and favorable to the candidate
- GPU memory usage remained effectively unchanged across datasets
- RSS differences were negligible
- train-time cost increased on some GPU points but stayed within a small
  practical range for this benchmark round

Decision:

- promote `AdamW + wd=5e-5` as the new default training configuration candidate
- keep scheduler disabled by default
- record CPU caution explicitly when reporting Stage B because CPU fairness was
  degraded by serial fallback in the restricted environment
