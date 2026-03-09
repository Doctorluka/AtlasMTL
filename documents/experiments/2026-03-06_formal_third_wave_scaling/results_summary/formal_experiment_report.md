# Formal experiment report

Date opened: `2026-03-07`

This document is the primary running report for the formal third-wave scaling
benchmark. Future dataset-level updates should be appended to this file instead
of creating a separate top-level report for each completed dataset.

## 1. Round scope and status

Formal dataset roster:

- `HLCA_Core`
- `PHMap_Lung_Full_v43_light`
- `mTCA`
- `DISCO_hPBMCs`
- `Vento` supplementary

Current round status:

- data audit / split / preprocessing: completed for all planned datasets
- manifest generation: completed
- formal execution completed at dataset level:
  - `HLCA_Core` main tracks completed
- formal execution still pending:
  - `PHMap_Lung_Full_v43_light`
  - `mTCA`
  - `DISCO_hPBMCs`
  - `Vento` supplementary

Current interpretation:

- `HLCA_Core` is the first dataset with a complete formal main-track result set
- the overall formal round is not complete yet
- the isolated CPU seurat track has now been narrowed to `10k / 20k / 30k / 50k`
  only for the formal round
- `Vento` remains on a supplementary reduced-ceiling track and is not mixed into
  the default main-panel execution queues
- within that restricted CPU seurat track, the earlier generic `60-minute`
  stop rule is no longer applied

## 2. Execution contract used

Build scaling:

- build sizes:
  - `10k / 20k / 30k / 50k / 100k / 150k / 200k / 300k`
- fixed query:
  - `build_eval_fixed_10k`

Predict scaling:

- fixed build:
  - `100k`
- predict sizes:
  - `1k / 3k / 5k / 8k / 10k / 15k / 20k / 50k`

Main methods currently reported in round-level comparison tables:

- CPU core:
  - `atlasmtl`
  - `reference_knn`
  - `celltypist`
  - `singler`
  - `symphony`
- GPU:
  - `atlasmtl`
  - `scanvi`

Methods handled in a separate isolated track:

- `seurat_anchor_transfer`

Reason:

- `seurat_anchor_transfer` is executed in an isolated CPU track because of its
  long-runtime behavior and should not block the main batch
- the formal round now limits CPU seurat to build scaling at
  `10k / 20k / 30k / 50k`
- retained CPU seurat predict scaling is now limited to `Vento`
  - `Vento`: fixed `build=50k`
  - non-`Vento` CPU seurat predict outputs are retained only as exploratory
    records
- after the restricted build policy was confirmed, the generic `60-minute`
  timeout was removed from the restricted CPU seurat track

## 3. Code paths

Round planning and protocol:

- `plan/2026-03-06_formal_third_wave_scaling_plan.md`
- `documents/protocols/formal_third_wave_scaling_protocol.md`
- `documents/protocols/third_wave_fairness_protocol.md`

Preparation and manifest generation:

- `documents/experiments/2026-03-06_formal_third_wave_scaling/scripts/prepare_formal_third_wave_scaling_inputs.py`
- `documents/experiments/2026-03-06_formal_third_wave_scaling/scripts/generate_formal_third_wave_manifests.py`

HLCA execution scripts used so far:

- `documents/experiments/2026-03-06_formal_third_wave_scaling/scripts/run_formal_hlca_cpu_core_main_batch.sh`
- `documents/experiments/2026-03-06_formal_third_wave_scaling/scripts/run_formal_hlca_gpu_main_batch.sh`
- `documents/experiments/2026-03-06_formal_third_wave_scaling/scripts/run_formal_hlca_cpu_seurat_main_batch.sh`

Shared runner:

- `documents/experiments/common/run_reference_heldout_scaleout_benchmark.py`

## 4. Result paths

Round prepared assets:

- `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/<dataset>/prepared/formal_split_v1/`

Current completed dataset result roots:

- `HLCA_Core`
  - CPU core:
    - `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/HLCA_Core/benchmark/formal_main/cpu_core/`
  - GPU:
    - `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/HLCA_Core/benchmark/formal_main/gpu/`
  - CPU seurat:
    - `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/HLCA_Core/benchmark/formal_main/cpu_seurat/`

Related repo-tracked notes:

- `documents/experiments/2026-03-06_formal_third_wave_scaling/results_summary/execution_report_2026-03-06_formal_hlca_sanity_start.md`
- `documents/experiments/2026-03-06_formal_third_wave_scaling/results_summary/hlca_formal_sanity_summary_2026-03-06.md`
- `documents/experiments/2026-03-06_formal_third_wave_scaling/results_summary/execution_report_2026-03-06_formal_hlca_main_batch_start.md`

## 5. Dataset section: HLCA_Core

### 5.1 Progress summary

| Track | Status | Completed points | Notes |
| --- | --- | --- | --- |
| `cpu_core` | completed | full build grid + full predict grid | all points succeeded |
| `gpu` | completed | full build grid + full predict grid | all points succeeded |
| `cpu_seurat` | completed under restricted formal scope | `build_10k / 20k / 30k / 50k -> eval10k` | restricted policy; not included in the main comparison tables below |

### 5.2 Build scaling summary

Representative points are shown below. Full per-point results remain in the tmp
result roots.

#### CPU core

| Point | Method | Accuracy | Macro-F1 | Train s | Predict s | Train RSS GB | Predict RSS GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `build_10k -> eval10k` | `atlasmtl` | 0.8390 | 0.6064 | 7.34 | 0.12 | 1.51 | 1.58 |
| `build_10k -> eval10k` | `celltypist` | 0.8543 | 0.6874 | 0.00 | 0.38 | 1.02 | 1.33 |
| `build_10k -> eval10k` | `singler` | 0.7907 | 0.7162 | 174.59 | 174.59 | 1.18 | 1.18 |
| `build_100k -> eval10k` | `atlasmtl` | 0.8713 | 0.7713 | 33.78 | 0.10 | 5.11 | 6.14 |
| `build_100k -> eval10k` | `celltypist` | 0.8682 | 0.7883 | 0.00 | 0.38 | 1.02 | 1.33 |
| `build_100k -> eval10k` | `singler` | 0.7983 | 0.7474 | 1617.04 | 1617.04 | 3.69 | 3.69 |
| `build_300k -> eval10k` | `atlasmtl` | 0.8844 | 0.8069 | 111.07 | 0.11 | 13.10 | 14.36 |
| `build_300k -> eval10k` | `celltypist` | 0.8802 | 0.8022 | 0.00 | 0.39 | 2.49 | 2.49 |
| `build_300k -> eval10k` | `singler` | 0.8018 | 0.7474 | 5040.37 | 5040.37 | 8.78 | 8.78 |

#### GPU

| Point | Method | Accuracy | Macro-F1 | Train s | Predict s | Train RSS GB | Predict RSS GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `build_10k -> eval10k` | `atlasmtl` | 0.8481 | 0.6405 | 1.63 | 0.08 | 1.93 | 1.95 |
| `build_10k -> eval10k` | `scanvi` | 0.8497 | 0.6525 | 32.70 | 14.46 | 2.16 | 2.16 |
| `build_100k -> eval10k` | `atlasmtl` | 0.8710 | 0.7616 | 8.42 | 0.11 | 5.57 | 6.81 |
| `build_100k -> eval10k` | `scanvi` | 0.8930 | 0.8097 | 312.56 | 14.17 | 3.66 | 3.66 |
| `build_300k -> eval10k` | `atlasmtl` | 0.8831 | 0.8054 | 28.41 | 0.06 | 13.61 | 16.05 |
| `build_300k -> eval10k` | `scanvi` | 0.9022 | 0.8368 | 857.55 | 14.57 | 6.99 | 6.99 |

### 5.3 Predict scaling summary

#### CPU core

| Point | Method | Accuracy | Macro-F1 | Train s | Predict s | Train RSS GB | Predict RSS GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `100k -> 1k` | `atlasmtl` | 0.8710 | 0.8492 | 30.32 | 0.01 | 5.04 | 5.12 |
| `100k -> 1k` | `celltypist` | 0.8740 | 0.8170 | 0.00 | 0.15 | 0.93 | 0.93 |
| `100k -> 10k` | `atlasmtl` | 0.8779 | 0.7578 | 36.64 | 0.13 | 5.11 | 5.24 |
| `100k -> 10k` | `celltypist` | 0.8644 | 0.8023 | 0.00 | 0.38 | 1.02 | 1.32 |
| `100k -> 50k` | `atlasmtl` | 0.8812 | 0.7750 | 36.85 | 0.69 | 5.41 | 5.49 |
| `100k -> 50k` | `celltypist` | 0.8691 | 0.7865 | 0.00 | 1.36 | 1.62 | 3.16 |

#### GPU

| Point | Method | Accuracy | Macro-F1 | Train s | Predict s | Train RSS GB | Predict RSS GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `100k -> 1k` | `atlasmtl` | 0.8670 | 0.8235 | 10.64 | 0.01 | 5.52 | 6.66 |
| `100k -> 1k` | `scanvi` | 0.8950 | 0.8631 | 303.56 | 1.66 | 3.61 | 3.61 |
| `100k -> 10k` | `atlasmtl` | 0.8729 | 0.7539 | 7.99 | 0.06 | 5.56 | 6.76 |
| `100k -> 10k` | `scanvi` | 0.8895 | 0.8315 | 300.20 | 14.66 | 3.67 | 3.67 |
| `100k -> 50k` | `atlasmtl` | 0.8725 | 0.7718 | 9.57 | 0.40 | 5.89 | 6.41 |
| `100k -> 50k` | `scanvi` | 0.8435 | 0.7189 | 318.93 | 71.22 | 3.97 | 3.97 |

### 5.4 HLCA current comparison and discussion

Current `HLCA` results support the following points.

1. `atlasmtl` is the most stable efficiency-oriented method across CPU and GPU.
   It keeps predict time extremely low while maintaining competitive accuracy
   and macro-F1.
2. `celltypist` is a strong CPU comparator. Its training cost is effectively
   negligible in this benchmark and it achieves the best CPU macro-F1 at many
   mid-to-large build and predict points.
3. `scanvi` is the strongest GPU comparator on most points when judged by pure
   performance, but its time cost is much larger than `atlasmtl`.
4. `singler` is accurate enough to remain competitive on some CPU points, but
   its runtime scales poorly and becomes one of the dominant CPU cost drivers.
5. On `HLCA`, larger build size generally helps `atlasmtl`, especially from
   `10k` to `100k` and again toward `300k`.
6. In GPU predict scaling, `scanvi` is usually best up to moderate query sizes,
   but the gap narrows at `50k`, where `atlasmtl` becomes the better
   performance-efficiency compromise and also exceeds `scanvi` on macro-F1.

### 5.5 HLCA seurat policy update

After reviewing the isolated CPU seurat results and runtime growth, the formal
round policy was tightened:

- keep CPU seurat only at `build_10k / 20k / 30k / 50k -> eval10k`
- do not continue to `100k+`
- keep CPU seurat build scaling restricted
- retain CPU seurat predict scaling only for `Vento`
- treat existing non-`Vento` CPU seurat predict results as exploratory only

Execution handling:

- the in-progress `build_100000_eval10k` and immediately-following
  `build_150000_eval10k` attempts were manually stopped once this narrowed
  policy was locked
- these stops should be interpreted as policy stops, not software failures

### 5.6 HLCA pending work

Still pending for `HLCA`:

- final integration of all `HLCA` results into round-level formal tables

## 6. Round-level pending work

### 6.1 Current dataset execution progress (`2026-03-08`)

Current formal execution status by dataset:

| Dataset | Status | Completed parts |
| --- | --- | --- |
| `HLCA_Core` | completed under current formal policy | `cpu_core` full build + full predict; `gpu` full build + full predict; `cpu_seurat` restricted `10k / 20k / 30k / 50k -> eval10k` |
| `PHMap_Lung_Full_v43_light` | in progress | `gpu` partial formal result set present; `cpu_seurat` restricted build track completed; `cpu_core` partial formal result set present and continuing from `predict_100000_20000` |
| `mTCA` | in progress | `gpu` partial formal result set present; `cpu_seurat` restricted build track completed through `50k -> eval10k`; `cpu_core` still pending first formal result point |
| `DISCO_hPBMCs` | in progress | `gpu` partial formal result set present; `cpu_seurat` started and continues after `build_10000_eval10k`; `cpu_core` still pending first formal result point |
| `Vento` | in progress on supplementary track | `gpu` result set complete; `cpu_seurat` build complete and retained predict scaling running on `50k` fixed build |

Round-level completion interpretation:

- fully completed dataset count: `1`
  - `HLCA_Core`
- datasets with at least one formal execution result: `4`
  - `HLCA_Core`
  - `PHMap_Lung_Full_v43_light`
  - `mTCA`
  - `DISCO_hPBMCs`

### 6.2 Active execution snapshot (`2026-03-08`)

At the time of this update, the active unfinished points were:

- `PHMap_Lung_Full_v43_light`
  - `cpu_core`
  - current point: `predict_100000_20000`
  - current dominant method process: `singler`
- `DISCO_hPBMCs`
  - `cpu_seurat`
  - current point: `build_20000_eval10k`
  - current dominant method process: `seurat_anchor_transfer`

### 6.3 Newly confirmed partial results outside HLCA

The following results were explicitly confirmed during the resumed execution
phase and should be treated as round-valid partial formal outputs:

| Dataset | Track | Point | Accuracy | Macro-F1 | Notes |
| --- | --- | --- | ---: | ---: | --- |
| `mTCA` | `cpu_seurat` | `build_50000 -> eval10k` | 0.8952 | 0.7745 | `TransferData-only`, fairness degraded by `joblib_serial_fallback` |
| `DISCO_hPBMCs` | `cpu_seurat` | `build_10000 -> eval10k` | 0.7879 | 0.5980 | `TransferData-only`, fairness degraded by `joblib_serial_fallback` |

### 6.4 Completed-output integrity check (`2026-03-08`)

A filesystem integrity scan was performed for all currently completed formal
points.

Checked artifacts:

- point-level:
  - `scaleout_status.json`
  - `runtime_manifest.yaml`
- method-level:
  - `summary.csv`
  - `stdout.log`
  - `stderr.log`

Current conclusion:

- no missing export files were detected for completed points
- completed points currently have structurally valid on-disk outputs across:
  - `HLCA_Core`
  - `PHMap_Lung_Full_v43_light`
  - `mTCA`
  - `DISCO_hPBMCs`
  - `Vento` supplementary
- remaining gaps are due to unfinished execution, not missing exports from
  completed runs

Interpretation notes:

- differences between `runtime_manifest` count and `scaleout_status` count
  should be read as in-progress or not-yet-finalized points
- `DISCO_hPBMCs cpu_core` still has no formal result root because that track
  has not yet started writing formal execution outputs

### 6.5 Seurat predict-scaling retention policy (`2026-03-08`)

Final formal decision:

- retain `seurat_anchor_transfer` build-scaling results for:
  - `HLCA_Core`
  - `PHMap_Lung_Full_v43_light`
  - `mTCA`
  - `DISCO_hPBMCs`
  - `Vento`
- retain `seurat_anchor_transfer` predict-scaling results in the formal
  comparison only for:
  - `Vento`
- store non-`Vento` seurat predict-scaling outputs as exploratory artifacts
  only

Reasoning:

- the restricted CPU seurat build policy already reflects a practical ceiling
  at `50k`

### 6.6 Post-benchmark paper-phase checklist

This checklist is the main follow-up framework after benchmark execution
finishes. The round should not be treated as paper-ready until these items are
closed.

#### A. Result freeze

- freeze the final retained method roster and dataset roster
- freeze which results are:
  - main-text
  - supplementary
  - exploratory-only
- explicitly close the round-level policy on:
  - `scanvi` GPU-only interpretation
  - `atlasmtl_cpu` vs `atlasmtl_gpu` split reporting
  - `seurat_anchor_transfer` restricted formal role

#### B. Integrity and reproducibility audit

- verify that every retained point has:
  - `scaleout_status.json`
  - `runtime_manifest.yaml`
  - `summary.csv`
  - `stdout.log`
  - `stderr.log`
- verify that retained point counts match intended manifest counts
- verify that all report-linked runtime roots still resolve correctly
- build a reproducibility index from formal tables back to runtime artifact
  roots

#### C. Scientific result review

- review anomalies in:
  - performance
  - runtime
  - RSS / GPU memory
  - fairness-degraded runs
- compare headline patterns against method papers or public expectations where
  possible
- decide per anomaly whether it is:
  - retained with explanation
  - rerun
  - supplementary-only
  - removed from headline comparison

#### D. Paper-facing output package

- generate the final round-level tables:
  - performance
  - resource
  - fairness / execution contract
  - dataset ceiling / completion
- generate the final round-level figures:
  - build-scaling curves
  - predict-scaling curves
  - performance-resource tradeoff plots
- mark each final table and figure as main-text or supplementary

#### E. Writing package

- write the benchmark methods text to exactly match the executed protocol
- write the results text with clear separation between:
  - headline findings
  - efficiency findings
  - supplementary comparator findings
  - exploratory findings
- record method-specific exceptions explicitly, including:
  - `Vento` supplementary reduced-ceiling handling
  - `seurat_anchor_transfer` restricted and partially exploratory handling

#### F. Final rerun gate

- only allow targeted reruns after the review above
- no broad pipeline redesign inside the formal round
- every rerun must state:
  - why it is needed
  - which table or figure it affects
  - where the updated execution note is recorded

Current implication:

- benchmark completion alone is not the end state
- the real closure condition is a paper-facing, audited, reproducible result
  package
- this is acceptable for build scaling because the missing `100k+` points are
  an explicit method ceiling rather than a silent protocol mismatch
- for predict scaling, comparing `seurat_anchor_transfer` under a lower
  practical reference ceiling against other methods under the shared `100k`
  reference contract would not be a fair main-table comparison
- `Vento` is the exception because its formal shared ceiling is already `50k`

Current seurat predict outputs on disk:

- retained formal-comparable:
  - `Vento`
    - `predict_50000_1000`
    - `predict_50000_3000`
    - `predict_50000_5000`
    - `predict_50000_8000`
    - `predict_50000_10000`
- exploratory-only:
  - `HLCA_Core`
    - `predict_100000_1000`
    - `predict_100000_3000`
    - `predict_100000_5000`
    - `predict_100000_8000`
    - `predict_100000_10000`
    - `predict_100000_15000`

- `PHMap_Lung_Full_v43_light`
- `mTCA`
- `DISCO_hPBMCs`
- `Vento` supplementary

### 6.7 Round comparative snapshot (`2026-03-09`)

The first round-level comparative snapshots have now been exported as
repo-tracked CSV files:

- `documents/experiments/2026-03-06_formal_third_wave_scaling/results_summary/formal_comparative_performance_snapshot_2026-03-09.csv`
- `documents/experiments/2026-03-06_formal_third_wave_scaling/results_summary/formal_comparative_resource_snapshot_2026-03-09.csv`
- `documents/experiments/2026-03-06_formal_third_wave_scaling/results_summary/formal_main_text_snapshot_2026-03-09.csv`
- `documents/experiments/2026-03-06_formal_third_wave_scaling/results_summary/formal_result_scope_map_2026-03-09.csv`

Scope used for these snapshots:

- main panel datasets:
  - `HLCA_Core`
  - `PHMap_Lung_Full_v43_light`
  - `mTCA`
  - `DISCO_hPBMCs`
- supplementary dataset:
  - `Vento`
- shared comparison points:
  - main-panel build: `build_100000 -> eval10k`
  - main-panel predict: `predict_100000 -> 10000`
  - `Vento` build: `build_50000 -> eval10k`
  - `Vento` predict: `predict_50000 -> 10000`
- `seurat_anchor_transfer`:
  - main-panel build-only supplementary comparator
  - `Vento` build + predict supplementary comparator

Interpretation boundary:

- these CSV files are intended to serve as the current paper-facing comparison
  snapshot
- non-`Vento` seurat predict outputs remain exploratory and are not included in
  the comparative snapshot tables
- `formal_main_text_snapshot_2026-03-09.csv` is the preferred source for the
  first draft of headline main-text tables
- `formal_result_scope_map_2026-03-09.csv` is the explicit retained-result map
  for deciding which rows stay in main text and which move to supplementary

### 6.8 Snapshot interpretation (`2026-03-09`)

Current headline interpretation from the comparative snapshot is as follows.

1. `scanvi` is the strongest pure-performance comparator on most GPU headline
   points.
   - it leads on `HLCA`, `PHMap`, `mTCA`, and `Vento`
   - `DISCO_hPBMCs` is the main exception where `atlasmtl` slightly exceeds
     `scanvi` on the retained `predict_100000 -> 10000` macro-F1 snapshot

2. `atlasmtl` is the strongest efficiency-oriented method across the round.
   - CPU and GPU predict time remain consistently low
   - GPU memory stays in a narrow and operationally cheap range, roughly
     `0.08-0.09 GB` peak for training and about `0.04-0.05 GB` for prediction
   - this efficiency advantage remains visible even when `scanvi` or
     `celltypist` outperform it on macro-F1

3. `celltypist` is a serious CPU baseline and should remain in headline
   comparisons.
   - on `mTCA`, its CPU macro-F1 exceeds `atlasmtl` at both the retained build
     and predict comparison points
   - on `HLCA` and `DISCO_hPBMCs`, it remains close enough to `atlasmtl` that
     it is not a throwaway baseline
   - its runtime cost is effectively negligible compared with `singler` and
     `seurat_anchor_transfer`

4. `PHMap_Lung_Full_v43_light` remains the hardest dataset in the current
   formal round.
   - all methods show materially lower performance than on `HLCA`, `mTCA`, and
     `DISCO_hPBMCs`
   - the gap is not specific to one method family, which suggests a real
     dataset-level difficulty rather than an isolated runner issue

5. `mTCA` and `Vento` are currently the strongest datasets for `atlasmtl`.
   - `mTCA` shows very strong performance across CPU and GPU, with
     `atlasmtl` already above `0.94` accuracy on retained headline points
   - `Vento` performs strongly under its reduced-ceiling contract, and the
     `atlasmtl` CPU/GPU gap is small relative to the overall performance level

6. `singler` and `seurat_anchor_transfer` are the dominant CPU cost drivers.
   - `singler` has acceptable accuracy on multiple datasets, but its runtime is
     orders of magnitude larger than `atlasmtl` and `celltypist`
   - `seurat_anchor_transfer` remains useful mainly as a documented
     supplementary comparator, not as an efficiency-competitive method

7. `Vento seurat` remains anomalous and should stay under caution.
   - the retained `Vento` build and predict results are preserved as formal
     records
   - however, their behavior is not fully consistent with expectation and
     should be reviewed against literature or external reports before deciding
     whether to display them in the main supplementary figures

Practical consequence for manuscript preparation:

- the current result pattern already supports a three-way narrative:
  - best absolute performance: usually `scanvi`
  - best performance-efficiency compromise: `atlasmtl`
  - strongest lightweight CPU baseline: `celltypist`

## 7. Append policy

Future updates should be appended to this report under dated dataset sections,
instead of creating a new parallel top-level report.

Append sections should include:

- newly completed points
- timeout or stop events
- updated summary tables if the result set changes materially
- discussion changes only when new results alter the current interpretation
