# Fourth-Round AtlasMTL Optimization Execution Plan

Date: **2026-03-09**  
Owner: `atlasmtl`  
Status: executable implementation plan

## 0) Purpose

This file converts the round-level strategy in
`plan/2026-03-09_fourth_round_atlasmtl_optimization_plan.md`
into an implementation sequence with concrete code touchpoints, scripts,
commands, runtime folders, and expected outputs.

This plan is still intentionally narrow:

- implement low-cost atlasmtl training controls only
- run atlasmtl-only representative-point checks
- keep the full third-wave formal benchmark locked as the comparison anchor

It does **not** reopen the full comparator grid.

## 1) Implementation outputs to create

Create or update the following code/assets in this round.

### Core code changes

1. Update `atlasmtl/core/train.py`
2. Update `benchmark/pipelines/run_benchmark.py`
3. Add or update focused tests under `tests/unit/` and `tests/integration/`

### Round-specific execution assets

1. Add `documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/scripts/generate_low_cost_optimization_manifests.py`
2. Add `documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/scripts/run_stage_a_cpu_core.sh`
3. Add `documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/scripts/run_stage_a_gpu.sh`
4. Add `documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/scripts/run_stage_b_cpu_core.sh`
5. Add `documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/scripts/run_stage_b_gpu.sh`
6. Add `documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/scripts/collect_stage_results.py`
7. Add `documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/scripts/freeze_baseline_table.py`

### Repo-tracked result files to populate during execution

1. `documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/results_summary/stage_a_baseline_anchor.csv`
2. `documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/results_summary/stage_a_screening_results.csv`
3. `documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/results_summary/stage_a_screening_results.md`
4. `documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/results_summary/stage_a_decision_note.md`
5. `documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/results_summary/stage_b_confirmation_results.csv`
6. `documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/results_summary/stage_b_confirmation_results.md`
7. `documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/results_summary/final_default_decision.md`

## 2) Core code changes

### 2.1 `atlasmtl/core/train.py`

Extend `build_model()` with explicit low-cost optimizer controls.

Add these parameters:

- `optimizer_name: str = "adam"`
- `weight_decay: float = 0.0`
- `scheduler_name: Optional[str] = None`
- `scheduler_factor: float = 0.5`
- `scheduler_patience: int = 5`
- `scheduler_min_lr: float = 1e-6`
- `scheduler_monitor: str = "val_loss"`

Implementation rules:

1. Support `optimizer_name in {"adam", "adamw"}` only.
2. Keep current behavior exactly when:
   - `optimizer_name="adam"`
   - `weight_decay=0.0`
   - `scheduler_name is None`
3. Use `torch.optim.AdamW` only when `optimizer_name="adamw"`.
4. Pass `weight_decay` directly into the optimizer.
5. Support `scheduler_name in {None, "reduce_lr_on_plateau"}` only.
6. Instantiate `ReduceLROnPlateau` only when validation is enabled.
7. If `scheduler_name="reduce_lr_on_plateau"` and `val_loader is None`, raise a
   clear `ValueError`.
8. Step the scheduler once per epoch after validation loss is computed.
9. Record scheduler/optimizer fields in `train_config`.
10. Record the final optimizer learning rate in `train_config`, for example via
    `final_learning_rate`.

Training metadata to add into `train_config`:

- `optimizer_name`
- `weight_decay`
- `scheduler_name`
- `scheduler_factor`
- `scheduler_patience`
- `scheduler_min_lr`
- `scheduler_monitor`
- `final_learning_rate`

### 2.2 `benchmark/pipelines/run_benchmark.py`

The atlasmtl runner already reads from `manifest["train"]`, but it currently
forwards only the existing fields. Extend the `build_model()` call so the
manifest can control:

- `optimizer_name`
- `weight_decay`
- `scheduler_name`
- `scheduler_factor`
- `scheduler_patience`
- `scheduler_min_lr`
- `scheduler_monitor`

The benchmark runner should stay permissive:

- if the fields are absent, it should use the baseline defaults
- no comparator method behavior should change in this round

### 2.3 Tests

Add focused tests rather than broad benchmark reruns.

Required tests:

1. Unit test: baseline defaults still serialize with
   `optimizer_name="adam"`, `weight_decay=0.0`, `scheduler_name=None`
2. Unit test: `optimizer_name="adamw"` is stored in `train_config`
3. Unit test: invalid `optimizer_name` raises `ValueError`
4. Unit test: invalid `scheduler_name` raises `ValueError`
5. Unit test: scheduler without validation split raises `ValueError`
6. Integration smoke test: `build_model(..., optimizer_name="adamw", weight_decay=1e-5, scheduler_name="reduce_lr_on_plateau", val_fraction=0.1, early_stopping_patience=5)` trains and returns expected metadata
7. Integration smoke test: benchmark manifest `train` block reaches atlasmtl and is reflected in `metrics.json`

## 3) Manifest generation design

Do not hand-edit dozens of manifests.

Instead, generate a small atlasmtl-only manifest subset from the already locked
third-wave manifest index:

- source index:
  `documents/experiments/2026-03-06_formal_third_wave_scaling/manifests/reference_heldout/manifest_index.json`
- source protocol assets:
  `documents/protocols/formal_third_wave_scaling_protocol.md`

### 3.1 Representative points to materialize

Generate manifests only for:

- `build_100000_eval10k`
- `predict_100000_10000`

### 3.2 Stage A datasets

- `PHMap_Lung_Full_v43_light`
- `mTCA`

### 3.3 Stage B datasets

- `HLCA_Core`
- `PHMap_Lung_Full_v43_light`
- `mTCA`
- `DISCO_hPBMCs`

### 3.4 Config variants

Stage A config names:

- `baseline`
- `adamw_wd_1e5`
- `adamw_wd_5e5`
- `adamw_wd_1e4`
- `adamw_bestwd_plateau`

Stage B config names:

- `baseline`
- `candidate_default`

### 3.5 Manifest-writing rules

The generation script should:

1. Load the formal third-wave manifest index.
2. Select only the requested datasets, track/device group, and representative point.
3. Copy the source manifest payload.
4. Restrict `methods` or downstream execution to `atlasmtl` only.
5. Preserve preprocessing, split, counts-layer, label-column, and predict
   settings exactly.
6. Overwrite only the `train` keys relevant to this round.
7. Add round metadata fields:
   - `experiment_round: "2026-03-09_atlasmtl_low_cost_optimization"`
   - `optimization_stage: "stage_a"` or `"stage_b"`
   - `config_name`
   - `source_formal_manifest`

Write outputs under:

- `documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/manifests/stage_a/`
- `documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/manifests/stage_b/`

Also write manifest indexes:

- `manifests/stage_a/manifest_index.json`
- `manifests/stage_b/manifest_index.json`

## 4) Baseline anchor extraction

Before running new jobs, create a frozen baseline table from the completed
third-wave results.

### 4.1 Input sources

- `documents/experiments/2026-03-06_formal_third_wave_scaling/results_summary/formal_comparative_performance_snapshot_2026-03-09.csv`
- `documents/experiments/2026-03-06_formal_third_wave_scaling/results_summary/formal_comparative_resource_snapshot_2026-03-09.csv`
- original per-run `metrics.json` from `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/` when needed for missing fields

### 4.2 Required baseline columns

- `dataset`
- `track`
- `point`
- `config_name`
- `accuracy`
- `macro_f1`
- `balanced_accuracy`
- `train_elapsed_seconds`
- `predict_elapsed_seconds`
- `train_process_peak_rss_gb`
- `predict_process_peak_rss_gb`
- `train_gpu_peak_memory_gb`
- `predict_gpu_peak_memory_gb`

Rules:

1. Use `config_name="baseline_anchor"`.
2. Do not backfill using new runs.
3. If a field is missing from the summary CSVs, load it from the original
   `metrics.json` and document that fallback in the script.

## 5) Stage A execution scripts

Stage A should run atlasmtl only.

### 5.1 CPU script behavior

`run_stage_a_cpu_core.sh` should:

1. Set fairness environment:
   - `OMP_NUM_THREADS=8`
   - `MKL_NUM_THREADS=8`
   - `OPENBLAS_NUM_THREADS=8`
   - `NUMEXPR_NUM_THREADS=8`
   - `NUMBA_CACHE_DIR=<repo>/.tmp/numba_cache`
   - `PYTHONPATH=<repo>`
   - `ATLASMTL_FAIRNESS_POLICY=cpu_only_strict`
2. Iterate only over Stage A CPU manifests.
3. For each manifest/config point, run:
   `documents/experiments/common/run_reference_heldout_scaleout_benchmark.py`
4. Use `--methods atlasmtl`.
5. Write outputs under:
   `/tmp/atlasmtl_benchmarks/2026-03-09/atlasmtl_low_cost_optimization/<dataset>/benchmark/stage_a/cpu_core/<point>/<config_name>/`
6. Skip already successful runs only if `metrics.json` and `summary.csv` both
   exist and `scaleout_status.json` reports atlasmtl success.

### 5.2 GPU script behavior

`run_stage_a_gpu.sh` should:

1. Reuse the same environment policy with GPU device checks.
2. Require a CUDA preflight before execution.
3. Run only Stage A GPU manifests.
4. Use `--device cuda --methods atlasmtl`.
5. Write outputs under:
   `/tmp/atlasmtl_benchmarks/2026-03-09/atlasmtl_low_cost_optimization/<dataset>/benchmark/stage_a/gpu/<point>/<config_name>/`

### 5.3 Stage A execution order

Run in this order:

1. `baseline` if a local reproduction is required
2. `adamw_wd_1e5`
3. `adamw_wd_5e5`
4. `adamw_wd_1e4`
5. aggregate interim results
6. choose `best_wd`
7. regenerate or activate `adamw_bestwd_plateau`
8. run scheduler check

Do not run the scheduler config before `best_wd` is chosen.

## 6) Stage B execution scripts

Stage B exists only if Stage A produces a candidate default.

`run_stage_b_cpu_core.sh` and `run_stage_b_gpu.sh` should match the Stage A
style but use:

- datasets:
  `HLCA_Core`, `PHMap_Lung_Full_v43_light`, `mTCA`, `DISCO_hPBMCs`
- configs:
  `baseline`, `candidate_default`
- output root:
  `/tmp/atlasmtl_benchmarks/2026-03-09/atlasmtl_low_cost_optimization/<dataset>/benchmark/stage_b/<track>/<point>/<config_name>/`

## 7) Aggregation and reporting

### 7.1 `collect_stage_results.py`

This script should read the per-run atlasmtl outputs and emit one flat table.

Read from each run directory:

- `scaleout_status.json`
- `runs/atlasmtl/metrics.json`
- `runs/atlasmtl/summary.csv`
- `runs/atlasmtl/stdout.log`
- `runs/atlasmtl/stderr.log`

Extract these fields:

- `dataset`
- `stage`
- `track`
- `point`
- `config_name`
- `optimizer_name`
- `weight_decay`
- `scheduler_name`
- `accuracy`
- `macro_f1`
- `balanced_accuracy`
- `train_elapsed_seconds`
- `predict_elapsed_seconds`
- `train_process_peak_rss_gb`
- `predict_process_peak_rss_gb`
- `train_gpu_peak_memory_gb`
- `predict_gpu_peak_memory_gb`
- `epochs_completed`
- `last_train_loss`
- `last_val_loss`
- `device_used`
- `num_threads_used`
- `runtime_fairness_degraded`
- `stdout_log`
- `stderr_log`

Derive one additional qualitative field:

- `early_stopping_note`

`early_stopping_note` may be derived using simple rules:

- `no_validation_split`
- `stopped_early`
- `ran_full_epochs`
- `scheduler_without_reduction_visible`
- `scheduler_reduced_lr`

The script does not need full log parsing if `train_config` already contains
enough metadata.

### 7.2 Markdown summaries

Generate concise markdown files from the CSV tables.

`stage_a_screening_results.md` should include:

1. baseline anchor table
2. Stage A result table
3. one short interpretation paragraph per track
4. explicit `best_wd` decision
5. explicit decision whether to run the scheduler check

`stage_b_confirmation_results.md` should include:

1. candidate vs baseline table
2. explicit keep/drop judgment
3. note on whether any benchmark rerun is justified

`final_default_decision.md` must end with exactly one explicit outcome:

- `keep baseline`
- `promote AdamW + weight_decay`
- `promote AdamW + weight_decay + ReduceLROnPlateau`

## 8) Concrete command sequence

### 8.1 Code validation before execution

```bash
python -m compileall atlasmtl benchmark
/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python -m pytest tests/unit -q
NUMBA_CACHE_DIR=/tmp/numba_cache /home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python -m pytest tests/integration/test_predict_integration.py -q
```

### 8.2 Generate round assets

```bash
/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python \
  documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/scripts/freeze_baseline_table.py

/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python \
  documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/scripts/generate_low_cost_optimization_manifests.py \
  --stage stage_a
```

### 8.3 Run Stage A

```bash
bash documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/scripts/run_stage_a_cpu_core.sh
bash documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/scripts/run_stage_a_gpu.sh
```

### 8.4 Aggregate Stage A and make decision

```bash
/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python \
  documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/scripts/collect_stage_results.py \
  --stage stage_a
```

If a candidate survives:

```bash
/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python \
  documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/scripts/generate_low_cost_optimization_manifests.py \
  --stage stage_b \
  --candidate-config <candidate_name>

bash documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/scripts/run_stage_b_cpu_core.sh
bash documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/scripts/run_stage_b_gpu.sh

/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python \
  documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/scripts/collect_stage_results.py \
  --stage stage_b
```

## 9) Decision logic to encode in the scripts

### 9.1 Best `weight_decay`

Choose `best_wd` only from the three AdamW runs.

Reject a `weight_decay` candidate if:

- it causes a meaningful `macro_f1` drop on `mTCA`
- it causes clear train-time blow-up
- it causes clear RSS or GPU-memory inflation
- it shows unstable convergence

Prefer the candidate that:

- is best or tied-best on `PHMap`
- is neutral on `mTCA`
- is operationally cheap

### 9.2 Scheduler keep/drop

Compare:

- `baseline`
- `best_wd`
- `best_wd + reduce_lr_on_plateau`

Keep the scheduler only if it improves practical default quality, not just one
isolated metric.

### 9.3 Stage B gate

Do not generate or run Stage B unless `stage_a_decision_note.md` explicitly
names a candidate default.

## 10) Stop conditions

Stop the round immediately if any of the following occurs:

1. baseline behavior changes when all new train parameters are left at defaults
2. `AdamW` path breaks benchmark serialization or model save/load
3. scheduler requires nontrivial runner redesign
4. Stage A rejects all `weight_decay` candidates
5. candidate runtime/memory penalty weakens atlasmtl practical positioning

If the round stops early, still write:

- `stage_a_baseline_anchor.csv`
- `stage_a_screening_results.csv` if partial runs exist
- `stage_a_decision_note.md`
- `final_default_decision.md`

## 11) Recommended implementation order inside the repo

Execute the development work in this order:

1. patch `atlasmtl/core/train.py`
2. patch `benchmark/pipelines/run_benchmark.py`
3. add unit/integration tests
4. run local validation
5. add baseline-freeze script
6. add manifest-generation script
7. add Stage A run scripts
8. run Stage A
9. add aggregation/report script if not already added before execution
10. make Stage A decision
11. run Stage B only if justified

This keeps the code change surface small and ensures the benchmark contract
stays locked while the training configuration becomes tunable.
