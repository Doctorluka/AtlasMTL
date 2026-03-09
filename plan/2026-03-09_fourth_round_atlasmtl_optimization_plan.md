# Fourth-Round AtlasMTL Optimization Plan

Date: **2026-03-09**  
Owner: `atlasmtl`  
Status: draft for review before execution

## 0) Goal

This round has one primary goal:

- improve `atlasmtl` with the lowest-cost training changes that have a realistic
  chance of improving robustness without changing the benchmark contract

This round has two secondary goals:

- define a minimal validation experiment that can confirm whether the new
  defaults are worth keeping
- decide explicitly whether any benchmark rerun is needed after the low-cost
  optimization is tested

This is **not** a new full benchmark round. The completed third-wave formal
benchmark remains the main comparator evidence base.

## 1) Locked scope for this round

### In scope

- optimizer regularization:
  - `AdamW`
  - `weight_decay`
- learning-rate schedule:
  - `ReduceLROnPlateau`
- compact result logging for the optimization sweep
- a small confirmatory rerun matrix if the new training settings look useful

### Explicitly out of scope

- changing the benchmark split contract
- changing the current formal third-wave dataset roster
- introducing new model heads
- introducing explicit hierarchical training losses
- turning hierarchy enforcement into the default benchmark behavior
- adding new comparator methods
- rerunning the full third-wave formal scaling grid

## 2) Current baseline to compare against

Use the currently locked `atlasmtl` defaults from the formal third-wave scaling
round and the atlasmtl parameter-lock round.

### Shared baseline

- `input_transform=binary`
- `max_epochs=50`
- `val_fraction=0.1`
- `early_stopping_patience=5`
- `early_stopping_min_delta=0.0`
- `reference_storage=external`
- `num_threads=8` for formal-style evaluation

### CPU baseline

- `learning_rate=3e-4`
- `hidden_sizes=[256,128]`
- `batch_size=128`
- optimizer: current `Adam`
- no weight decay
- no scheduler

### GPU baseline

- `learning_rate=1e-3`
- `hidden_sizes=[1024,512]`
- `batch_size=512`
- optimizer: current `Adam`
- no weight decay
- no scheduler

### Important note

Do not change:

- preprocessing
- group-aware split assets
- dataset manifests used for point selection
- comparator settings

The only intended behavioral change in this round is the `atlasmtl` training
configuration.

## 3) Datasets and benchmark points to use

This round uses a two-stage design.

### Stage A: low-cost screening

Use only these datasets:

- `PHMap_Lung_Full_v43_light`
- `mTCA`

Reason:

- `PHMap` is the difficult dataset and best stress-test for stability
- `mTCA` is the strong-performing dataset and best check for regression risk

Use only these representative points:

- build-scaling representative point:
  - `build_100000_eval10k`
- predict-scaling representative point:
  - `predict_100000_10000`

Run both `cpu_core` and `gpu` tracks if the point already exists and is
supported by the dataset/method contract.

### Stage B: confirmatory expansion

Only if Stage A identifies a clearly better low-cost configuration, extend to:

- `HLCA_Core`
- `PHMap_Lung_Full_v43_light`
- `mTCA`
- `DISCO_hPBMCs`

Use the same two representative points only:

- `build_100000_eval10k`
- `predict_100000_10000`

Do not expand back to the full scaling grid in this round.

## 4) Training configurations to test

### Stage A matrix

For each device track, run exactly these configurations:

1. `baseline`
2. `baseline + wd=1e-5`
3. `baseline + wd=5e-5`
4. `baseline + wd=1e-4`
5. `best_wd + scheduler`

### Optimizer and scheduler details

#### Weight-decay runs

- replace `torch.optim.Adam` with `torch.optim.AdamW`
- keep the current learning rate for the device track
- set:
  - `weight_decay=1e-5`
  - `weight_decay=5e-5`
  - `weight_decay=1e-4`

#### Scheduler run

Use:

- scheduler: `ReduceLROnPlateau`
- monitor: validation loss
- factor: `0.5`
- patience: `5`
- min_lr: `1e-6`

Apply the scheduler only in configuration `5`, using the best `weight_decay`
value selected from configurations `2` to `4`.

### Locked rule

Do not test:

- more than one scheduler type
- more than three `weight_decay` values
- mixed grids of `lr x wd x scheduler`

This round is meant to decide whether a low-cost enhancement is worth keeping,
not to open a large hyperparameter search.

## 5) Execution order

### Step 1: freeze a reference baseline table

Before any new run, record the existing baseline values for the chosen points
from the completed third-wave benchmark:

- dataset
- track
- point
- accuracy
- macro_f1
- train_elapsed_seconds
- predict_elapsed_seconds
- train_process_peak_rss_gb
- predict_process_peak_rss_gb
- train_gpu_peak_memory_gb
- predict_gpu_peak_memory_gb

This table becomes the immutable comparison anchor for this round.

### Step 2: run Stage A weight-decay screening

For both `PHMap` and `mTCA`, and for each eligible `cpu_core` / `gpu` track:

- run baseline reproduction once if needed for environment consistency
- run the three `AdamW + weight_decay` variants
- collect the same metrics as the baseline table
- collect:
  - `epochs_completed`
  - `last_train_loss`
  - `last_val_loss`
  - any early-stopping behavior note from logs

### Step 3: choose one `weight_decay`

Choose the candidate that best satisfies all of the following:

- no meaningful `macro_f1` drop on `mTCA`
- best or tied-best `macro_f1` on `PHMap`
- no material train-time blow-up
- no material peak-memory increase
- no sign of unstable training or erratic early stopping

If no `weight_decay` candidate satisfies these conditions, stop the round and
keep the current baseline.

### Step 4: run the scheduler check

Using the selected `weight_decay` only:

- add `ReduceLROnPlateau`
- rerun the same Stage A dataset/track/point matrix
- compare only against:
  - baseline
  - best `weight_decay` without scheduler

### Step 5: make the keep-or-drop decision

Promote the new training default only if the scheduler-enhanced configuration
meets all of the following:

- `macro_f1` is not worse on `mTCA`
- `macro_f1` is at least neutral and preferably better on `PHMap`
- runtime increase is minor
- memory increase is minor
- epochs completed or validation-loss behavior suggests more stable convergence

If the scheduler does not help, keep `AdamW + best_weight_decay` only as the
candidate default.

### Step 6: run Stage B confirmation only if justified

Stage B is allowed only if Step 5 identifies a candidate default worth keeping.

Stage B reruns only these four datasets:

- `HLCA_Core`
- `PHMap_Lung_Full_v43_light`
- `mTCA`
- `DISCO_hPBMCs`

at only these points:

- `build_100000_eval10k`
- `predict_100000_10000`

and only for:

- old baseline
- new candidate default

## 6) Metrics to record and decision rules

### Required metrics

Record these fields for every run:

- `dataset`
- `track`
- `point`
- `config_name`
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

### Main ranking metric

Use `macro_f1` as the primary decision metric.

### Secondary metrics

Use these only as constraints:

- `accuracy`
- runtime
- peak RSS
- GPU peak memory

### Decision thresholds

Use the following qualitative thresholds:

- keep candidate if `macro_f1` is neutral-to-better overall
- reject candidate if it harms `mTCA` while only giving trivial `PHMap` gain
- reject candidate if it increases runtime or memory enough to weaken the
  practical positioning of `atlasmtl`

### Practical interpretation rule

The new default does **not** need to win every dataset. It only needs to be a
better practical default for the framework overall.

## 7) Benchmark rerun policy

### Do not rerun by default

Do **not** rerun the completed third-wave formal scaling grid in this round.

Reason:

- the formal round already answers the benchmark-scale questions
- the current round only tests whether a low-cost atlasmtl training change is
  worth promoting

### When a rerun is justified

Only do a targeted rerun if one of these conditions is met:

1. Stage B confirms a better default and a paper-facing comparison update is
   needed for representative points
2. a later hierarchy secondary analysis needs per-cell predictions that were
   not retained in the original run directories

### Allowed rerun scope

If reruns are needed, rerun only:

- representative points
- selected datasets
- atlasmtl runs only, unless a cross-method mapped-coarse analysis explicitly
  requires comparator predictions

Do not reopen the full comparator grid unless a new formal question is being
asked.

## 8) Record locations for this round

### Plan

- `plan/2026-03-09_fourth_round_atlasmtl_optimization_plan.md`

### Experiment dossier root

- `documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/`

### Repo-tracked notes and summaries

- `documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/README.md`
- `documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/results_summary/`

### Runtime outputs

Use one dedicated tmp root:

- `/tmp/atlasmtl_benchmarks/2026-03-09/atlasmtl_low_cost_optimization/`

Inside that root, keep:

- per-dataset run folders
- raw logs
- per-run `metrics.json`
- per-run `summary.csv`
- any temporary aggregation outputs

### Recommended summary files

When execution begins, keep these repo-tracked outputs:

- `results_summary/stage_a_screening_results.csv`
- `results_summary/stage_a_screening_results.md`
- `results_summary/stage_a_decision_note.md`
- `results_summary/stage_b_confirmation_results.csv`
- `results_summary/stage_b_confirmation_results.md`
- `results_summary/final_default_decision.md`

## 9) Execution checklist

Execute this round in order.

1. Create the tmp output root for this round.
2. Copy the locked baseline metrics for the selected representative points into
   a baseline comparison table.
3. Implement the training changes behind explicit config fields:
   - optimizer choice
   - weight decay
   - scheduler choice
4. Run Stage A on `PHMap` and `mTCA`.
5. Produce `stage_a_screening_results.csv` and a short interpretation note.
6. Decide:
   - reject all changes and stop, or
   - choose one candidate default
7. If a candidate survives, run Stage B confirmation on the four main-panel
   datasets.
8. Produce `stage_b_confirmation_results.csv`.
9. Write `final_default_decision.md` with one explicit outcome:
   - keep baseline
   - promote `AdamW + wd`
   - promote `AdamW + wd + scheduler`

## 10) Success criteria

This round is successful if it produces a clear decision on the default
training configuration without needing a full benchmark rerun.

Accepted successful outcomes:

- evidence supports keeping the current baseline
- evidence supports promoting `AdamW + best_weight_decay`
- evidence supports promoting `AdamW + best_weight_decay + scheduler`

Unsuccessful outcome:

- the round ends without a decision because the screening matrix or record
  keeping was under-specified

This plan is written to avoid that failure mode.
