# Second-Wave Round Status (`2026-03-04`)

This file is the current repo-tracked status summary for the second-wave
reference-heldout scale-out round.

It replaces stale interpretations that were scattered across earlier per-dataset
notes. Status below is based on both:

- repo-tracked dossiers under `documents/experiments/`
- actual runtime artifacts under `~/tmp/atlasmtl_benchmarks/2026-03-04/`

If a run is not backed by either a repo-tracked execution record or a concrete
runtime artifact path, it is not counted here as completed.

## Scope

Round objective:

- expand from first-wave smoke runs to larger reference-heldout scale-out runs
- default target: `100k` build + `10k` heldout + nested `5k`
- reduced `Vento` target: `50k` build + `10k` heldout + nested `5k` when feasible

Shared execution/protocol files:

- `plan/2026-03-04_second_wave_scaleout_benchmark_plan.md`
- `documents/protocols/reference_heldout_scaleout_execution.md`
- `documents/experiments/common/prepare_reference_heldout_scaleout.py`
- `documents/experiments/common/run_reference_heldout_scaleout_benchmark.py`

## Current status at a glance

| Dataset | Status | Evidence level | Notes |
| --- | --- | --- | --- |
| `PHMap_Lung_Full_v43_light` | completed | repo reports + tmp outputs | `100k/10k/5k` prepared, `10k` benchmark finished |
| `DISCO_hPBMCs` | completed | repo reports + tmp outputs | `100k/10k/5k` prepared, `10k` benchmark finished |
| `mTCA` | completed after correction | repo reports + tmp outputs | first run invalid due to wrong species; rerun succeeded |
| `HLCA_Core` | completed after recovery | repo reports + tmp outputs | `100k/10k/5k` prepared, `10k` benchmark recovered to `7/7` success after interruption |
| `cd4` | excluded from current round | dossier + prep log only | lacks acceptable raw counts for the current formal preprocessing path |
| `cd8` | excluded from current round | dossier only | lacks acceptable raw counts for the current formal preprocessing path |
| `Vento` | completed | repo reports + tmp outputs | `50k/10k/5k` prepared, `10k` and nested `5k` benchmarks both finished |

## Shared process

### Preparation flow

All second-wave preparation is intended to use:

- `documents/experiments/common/prepare_reference_heldout_scaleout.py`

Dataset wrapper scripts:

- `documents/experiments/2026-03-03_phmap_benchmark/scripts/run_prepare_scaleout.sh`
- `documents/experiments/2026-03-03_projectsvr_disco_benchmark/scripts/run_prepare_scaleout.sh`
- `documents/experiments/2026-03-03_hlca_benchmark/scripts/run_prepare_scaleout.sh`
- `documents/experiments/2026-03-04_projectsvr_mtca_benchmark/scripts/run_prepare_scaleout.sh`
- `documents/experiments/2026-03-04_projectsvr_cd4_benchmark/scripts/run_prepare_scaleout.sh`
- `documents/experiments/2026-03-04_projectsvr_cd8_benchmark/scripts/run_prepare_scaleout.sh`
- `documents/experiments/2026-03-04_projectsvr_vento_benchmark/scripts/run_prepare_scaleout.sh`

Expected prepared outputs per dataset:

- `reference_train.h5ad`
- `heldout_test_10k.h5ad`
- `heldout_test_5k.h5ad`
- `feature_panel.json`
- `split_plan.json`
- `split_summary.json`
- `preprocessing_summary.json`
- `preparation_resource_summary.json`

### Benchmark flow

All completed second-wave benchmark runs use:

- `documents/experiments/common/run_reference_heldout_scaleout_benchmark.py`

Expected benchmark-side outputs per completed run:

- `scaleout_status.json`
- `runtime_manifest.yaml`
- `runtime_manifest_celltypist.yaml`
- `runs/<method>/summary.csv`
- `runs/<method>/metrics.json`

## Dataset-by-dataset record

### 1) `PHMap_Lung_Full_v43_light`

Status:

- completed and backed by both repo reports and tmp runtime outputs

Repo-tracked files:

- dossier root:
  `documents/experiments/2026-03-03_phmap_benchmark/`
- preparation report:
  `documents/experiments/2026-03-03_phmap_benchmark/results_summary/execution_report_2026-03-04_scaleout_preparation.md`
- preparation record:
  `documents/experiments/2026-03-03_phmap_benchmark/results_summary/experiment_record_2026-03-04_scaleout_preparation.md`
- benchmark report:
  `documents/experiments/2026-03-03_phmap_benchmark/results_summary/execution_report_2026-03-04_scaleout_benchmark_10k.md`
- benchmark record:
  `documents/experiments/2026-03-03_phmap_benchmark/results_summary/experiment_record_2026-03-04_scaleout_benchmark_10k.md`
- prep manifest:
  `documents/experiments/2026-03-03_phmap_benchmark/manifests/reference_heldout/PHMap_Lung_Full_v43_light__anno_lv4__scaleout_prep_v1.yaml`
- runtime manifest `10k`:
  `documents/experiments/2026-03-03_phmap_benchmark/manifests/reference_heldout/PHMap_Lung_Full_v43_light__anno_lv4__scaleout_runtime_10k_v1.yaml`
- runtime manifest `5k`:
  `documents/experiments/2026-03-03_phmap_benchmark/manifests/reference_heldout/PHMap_Lung_Full_v43_light__anno_lv4__scaleout_runtime_5k_v1.yaml`

Runtime artifact roots:

- prepared root:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/PHMap_Lung_Full_v43_light/prepared/group_split_v2_train100k_test10k_nested5k/`
- benchmark root:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/PHMap_Lung_Full_v43_light/benchmark/group_split_v2_train100k_test10k/all_methods_v1/`
- preparation resource summary:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/PHMap_Lung_Full_v43_light/prepared/group_split_v2_train100k_test10k_nested5k/preparation_resource_summary.json`
- split summary:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/PHMap_Lung_Full_v43_light/prepared/group_split_v2_train100k_test10k_nested5k/split_summary.json`
- benchmark status:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/PHMap_Lung_Full_v43_light/benchmark/group_split_v2_train100k_test10k/all_methods_v1/scaleout_status.json`

Recorded process:

- first-wave preparation and smoke had already been completed in the same
  dossier on `2026-03-03`
- second-wave preparation succeeded at `100k/10k/5k`
- second-wave `10k` benchmark completed with `7/7` methods successful

Recorded issues and resolutions:

- no blocking code change was required during the `10k` benchmark run
- the main runtime burden came from slower external comparators
- external-comparator RSS accounting remains incomplete for several methods;
  this is a monitoring gap, not a run failure

### 2) `DISCO_hPBMCs`

Status:

- completed and backed by both repo reports and tmp runtime outputs

Repo-tracked files:

- dossier root:
  `documents/experiments/2026-03-03_projectsvr_disco_benchmark/`
- preparation report:
  `documents/experiments/2026-03-03_projectsvr_disco_benchmark/results_summary/execution_report_2026-03-04_scaleout_preparation.md`
- preparation record:
  `documents/experiments/2026-03-03_projectsvr_disco_benchmark/results_summary/experiment_record_2026-03-04_scaleout_preparation.md`
- benchmark report:
  `documents/experiments/2026-03-03_projectsvr_disco_benchmark/results_summary/execution_report_2026-03-04_scaleout_benchmark_10k.md`
- benchmark record:
  `documents/experiments/2026-03-03_projectsvr_disco_benchmark/results_summary/experiment_record_2026-03-04_scaleout_benchmark_10k.md`
- prep manifest:
  `documents/experiments/2026-03-03_projectsvr_disco_benchmark/manifests/reference_heldout/DISCO_hPBMCs__cell_subtype__scaleout_prep_v1.yaml`
- runtime manifest `10k`:
  `documents/experiments/2026-03-03_projectsvr_disco_benchmark/manifests/reference_heldout/DISCO_hPBMCs__cell_subtype__scaleout_runtime_10k_v1.yaml`
- runtime manifest `5k`:
  `documents/experiments/2026-03-03_projectsvr_disco_benchmark/manifests/reference_heldout/DISCO_hPBMCs__cell_subtype__scaleout_runtime_5k_v1.yaml`

Runtime artifact roots:

- prepared root:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/DISCO_hPBMCs/prepared/group_split_v2_train100k_test10k_nested5k/`
- benchmark root:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/DISCO_hPBMCs/benchmark/group_split_v2_train100k_test10k/all_methods_v2/`
- preparation resource summary:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/DISCO_hPBMCs/prepared/group_split_v2_train100k_test10k_nested5k/preparation_resource_summary.json`
- split summary:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/DISCO_hPBMCs/prepared/group_split_v2_train100k_test10k_nested5k/split_summary.json`
- benchmark status:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/DISCO_hPBMCs/benchmark/group_split_v2_train100k_test10k/all_methods_v2/scaleout_status.json`

Recorded process:

- first-wave preparation and smoke had already been completed in the same
  dossier on `2026-03-03`
- second-wave preparation succeeded at `100k/10k/5k`
- second-wave `10k` benchmark completed with `7/7` methods successful

Recorded issues and resolutions:

- the generic benchmark runner could not launch `celltypist` directly because a
  comparator model path had to be created first
- this was resolved by adding the shared wrapper:
  `documents/experiments/common/run_reference_heldout_scaleout_benchmark.py`
- that wrapper now prepares comparator-specific log1p inputs, trains the
  CellTypist model, writes `runtime_manifest_celltypist.yaml`, and then runs
  the standard benchmark pipeline
- `seurat_anchor_transfer` remained stable only on the `TransferData`-style
  backend; this is recorded in the benchmark report

Important data-contract caveat:

- `data_registry/reference_data_inventory_2026-03-03.md` now records that
  `DISCO_hPBMCs` was re-checked on `2026-03-04` and that `adata.X` looks more
  like non-integer positive log-normalized values rather than strict raw counts
- the engineering benchmark run exists and completed, but that newer inventory
  note should be treated as the stricter contract interpretation for future
  formal runs

### 3) `mTCA`

Status:

- completed after a recorded failure-and-correction cycle

Repo-tracked files:

- dossier root:
  `documents/experiments/2026-03-04_projectsvr_mtca_benchmark/`
- benchmark report:
  `documents/experiments/2026-03-04_projectsvr_mtca_benchmark/results_summary/execution_report_2026-03-04_scaleout_benchmark_10k.md`
- benchmark record:
  `documents/experiments/2026-03-04_projectsvr_mtca_benchmark/results_summary/experiment_record_2026-03-04_scaleout_benchmark_10k.md`
- prep manifest:
  `documents/experiments/2026-03-04_projectsvr_mtca_benchmark/manifests/reference_heldout/mTCA__Cell_type_level3__scaleout_prep_v1.yaml`
- runtime manifest `10k`:
  `documents/experiments/2026-03-04_projectsvr_mtca_benchmark/manifests/reference_heldout/mTCA__Cell_type_level3__scaleout_runtime_10k_v1.yaml`
- runtime manifest `5k`:
  `documents/experiments/2026-03-04_projectsvr_mtca_benchmark/manifests/reference_heldout/mTCA__Cell_type_level3__scaleout_runtime_5k_v1.yaml`
- prep wrapper:
  `documents/experiments/2026-03-04_projectsvr_mtca_benchmark/scripts/run_prepare_scaleout.sh`

Runtime artifact roots:

- prepared root:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/mTCA/prepared/group_split_v2_train100k_test10k_nested5k/`
- invalid first benchmark root:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/mTCA/benchmark/group_split_v2_train100k_test10k/all_methods_v1/`
- corrected benchmark root:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/mTCA/benchmark/group_split_v2_train100k_test10k/all_methods_v2_mouse_fix/`
- preparation resource summary:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/mTCA/prepared/group_split_v2_train100k_test10k_nested5k/preparation_resource_summary.json`
- split summary:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/mTCA/prepared/group_split_v2_train100k_test10k_nested5k/split_summary.json`
- invalid first benchmark status:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/mTCA/benchmark/group_split_v2_train100k_test10k/all_methods_v1/scaleout_status.json`
- corrected benchmark status:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/mTCA/benchmark/group_split_v2_train100k_test10k/all_methods_v2_mouse_fix/scaleout_status.json`

Recorded process:

- second-wave preparation was attempted with the `mTCA` dossier
- the first benchmark attempt was invalid and should not be interpreted
- the prep manifest was corrected
- preparation and benchmark were rerun from scratch
- the corrected `10k` benchmark completed with `7/7` methods successful

Recorded failure and correction:

- root cause:
  the prep manifest used `species: human` for a mouse dataset
- recorded correction:
  `documents/experiments/2026-03-04_projectsvr_mtca_benchmark/results_summary/experiment_record_2026-03-04_scaleout_benchmark_10k.md`
  explicitly states that
  `documents/experiments/2026-03-04_projectsvr_mtca_benchmark/manifests/reference_heldout/mTCA__Cell_type_level3__scaleout_prep_v1.yaml`
  was changed from `species: human` to `species: mouse`
- effect of correction:
  the corrected run recovered from an invalid `19`-gene feature space to a
  valid `3000 HVG` reference panel and all comparators became runnable

Remaining caveats:

- current `celltypist` result uses the wrapped logistic-regression backend
- several R-based comparators still have incomplete RSS accounting

### 4) `HLCA_Core`

Status:

- completed after interruption recovery and now backed by repo reports plus tmp
  runtime outputs

Repo-tracked files:

- dossier root:
  `documents/experiments/2026-03-03_hlca_benchmark/`
- README:
  `documents/experiments/2026-03-03_hlca_benchmark/README.md`
- checklist:
  `documents/experiments/2026-03-03_hlca_benchmark/plan/execution_checklist.md`
- prep manifest:
  `documents/experiments/2026-03-03_hlca_benchmark/manifests/reference_heldout/HLCA_Core__ann_level_5__scaleout_prep_v1.yaml`
- runtime manifest `10k`:
  `documents/experiments/2026-03-03_hlca_benchmark/manifests/reference_heldout/HLCA_Core__ann_level_5__scaleout_runtime_10k_v1.yaml`
- runtime manifest `5k`:
  `documents/experiments/2026-03-03_hlca_benchmark/manifests/reference_heldout/HLCA_Core__ann_level_5__scaleout_runtime_5k_v1.yaml`
- preparation report:
  `documents/experiments/2026-03-03_hlca_benchmark/results_summary/execution_report_2026-03-05_scaleout_preparation.md`
- preparation record:
  `documents/experiments/2026-03-03_hlca_benchmark/results_summary/experiment_record_2026-03-05_scaleout_preparation.md`
- benchmark report:
  `documents/experiments/2026-03-03_hlca_benchmark/results_summary/execution_report_2026-03-05_scaleout_benchmark_10k.md`
- benchmark record:
  `documents/experiments/2026-03-03_hlca_benchmark/results_summary/experiment_record_2026-03-05_scaleout_benchmark_10k.md`

Runtime artifact roots:

- prepared root:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/prepared/group_split_v2_train100k_test10k_nested5k/`
- benchmark root:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/benchmark/group_split_v2_train100k_test10k/all_methods_v1/`
- preparation resource summary:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/prepared/group_split_v2_train100k_test10k_nested5k/preparation_resource_summary.json`
- split summary:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/prepared/group_split_v2_train100k_test10k_nested5k/split_summary.json`
- benchmark status:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/benchmark/group_split_v2_train100k_test10k/all_methods_v1/scaleout_status.json`

Recorded process:

- second-wave preparation was materialized at `100k/10k/5k`
- all-method benchmark execution was interrupted before final status
  materialization
- `seurat_anchor_transfer` was rerun separately on the same runtime manifest
- top-level `scaleout_status.json` was reconstructed from per-method outputs
  after verification
- final state is `7/7` successful methods

Recorded failure and correction:

- failure:
  wrapper interruption caused missing top-level status manifest
- correction:
  single-method rerun for `seurat_anchor_transfer` plus manual status
  reconstruction
- audit anchor:
  `documents/experiments/2026-03-03_hlca_benchmark/results_summary/experiment_record_2026-03-05_scaleout_benchmark_10k.md`

### 5) `cd4`

Status:

- excluded from the current formal round because the dataset does not satisfy
  the current raw-count contract

Repo-tracked files:

- dossier root:
  `documents/experiments/2026-03-04_projectsvr_cd4_benchmark/`
- README:
  `documents/experiments/2026-03-04_projectsvr_cd4_benchmark/README.md`
- checklist:
  `documents/experiments/2026-03-04_projectsvr_cd4_benchmark/plan/execution_checklist.md`
- prep manifest:
  `documents/experiments/2026-03-04_projectsvr_cd4_benchmark/manifests/reference_heldout/cd4__cell_subtype__scaleout_prep_v1.yaml`
- runtime manifest `10k`:
  `documents/experiments/2026-03-04_projectsvr_cd4_benchmark/manifests/reference_heldout/cd4__cell_subtype__scaleout_runtime_10k_v1.yaml`
- runtime manifest `5k`:
  `documents/experiments/2026-03-04_projectsvr_cd4_benchmark/manifests/reference_heldout/cd4__cell_subtype__scaleout_runtime_5k_v1.yaml`
- prep wrapper:
  `documents/experiments/2026-03-04_projectsvr_cd4_benchmark/scripts/run_prepare_scaleout.sh`

Discovered runtime-side evidence:

- log:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/cd4/logs/cd4_scaleout_prep.log`
- pid file:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/cd4/logs/cd4_scaleout_prep.pid`

What is missing:

- no prepared output directory with `reference_train.h5ad`
- no `split_summary.json`
- no `preparation_resource_summary.json`
- no benchmark output directory
- no repo-tracked execution report or experiment record

Interpretation:

- `cd4` should not be counted as a completed benchmark dataset for this round
- the current dossier and log only show that preparation machinery was staged
- per current project interpretation, `cd4` is excluded until an acceptable raw
  counts source is available

### 6) `cd8`

Status:

- excluded from the current formal round because the dataset does not satisfy
  the current raw-count contract

Repo-tracked files:

- dossier root:
  `documents/experiments/2026-03-04_projectsvr_cd8_benchmark/`
- README:
  `documents/experiments/2026-03-04_projectsvr_cd8_benchmark/README.md`
- checklist:
  `documents/experiments/2026-03-04_projectsvr_cd8_benchmark/plan/execution_checklist.md`
- prep manifest:
  `documents/experiments/2026-03-04_projectsvr_cd8_benchmark/manifests/reference_heldout/cd8__cell_subtype__scaleout_prep_v1.yaml`
- runtime manifest `10k`:
  `documents/experiments/2026-03-04_projectsvr_cd8_benchmark/manifests/reference_heldout/cd8__cell_subtype__scaleout_runtime_10k_v1.yaml`
- runtime manifest `5k`:
  `documents/experiments/2026-03-04_projectsvr_cd8_benchmark/manifests/reference_heldout/cd8__cell_subtype__scaleout_runtime_5k_v1.yaml`
- prep wrapper:
  `documents/experiments/2026-03-04_projectsvr_cd8_benchmark/scripts/run_prepare_scaleout.sh`

What is missing:

- no discovered tmp output root under
  `~/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/cd8/`
- no repo-tracked execution report
- no repo-tracked experiment record

Interpretation:

- `cd8` should not be counted as a completed benchmark dataset for this round
- the current dossier only shows that the scale-out path was scaffolded
- per current project interpretation, `cd8` is excluded until an acceptable raw
  counts source is available

### 7) `Vento`

Status:

- completed for the second-wave benchmark scope and backed by repo records plus
  tmp runtime outputs

Repo-tracked files:

- dossier root:
  `documents/experiments/2026-03-04_projectsvr_vento_benchmark/`
- README:
  `documents/experiments/2026-03-04_projectsvr_vento_benchmark/README.md`
- checklist:
  `documents/experiments/2026-03-04_projectsvr_vento_benchmark/plan/execution_checklist.md`
- prep manifest:
  `documents/experiments/2026-03-04_projectsvr_vento_benchmark/manifests/reference_heldout/Vento__annotation__scaleout_prep_v1.yaml`
- runtime manifest `10k`:
  `documents/experiments/2026-03-04_projectsvr_vento_benchmark/manifests/reference_heldout/Vento__annotation__scaleout_runtime_10k_v1.yaml`
- runtime manifest `5k`:
  `documents/experiments/2026-03-04_projectsvr_vento_benchmark/manifests/reference_heldout/Vento__annotation__scaleout_runtime_5k_v1.yaml`
- prep wrapper:
  `documents/experiments/2026-03-04_projectsvr_vento_benchmark/scripts/run_prepare_scaleout.sh`
- benchmark report (`10k`):
  `documents/experiments/2026-03-04_projectsvr_vento_benchmark/results_summary/execution_report_2026-03-04_scaleout_benchmark_10k.md`
- benchmark record (`10k`):
  `documents/experiments/2026-03-04_projectsvr_vento_benchmark/results_summary/experiment_record_2026-03-04_scaleout_benchmark_10k.md`
- benchmark report (`5k` nested):
  `documents/experiments/2026-03-04_projectsvr_vento_benchmark/results_summary/execution_report_2026-03-04_scaleout_benchmark_5k.md`
- benchmark record (`5k` nested):
  `documents/experiments/2026-03-04_projectsvr_vento_benchmark/results_summary/experiment_record_2026-03-04_scaleout_benchmark_5k.md`

Runtime artifact roots:

- prepared root:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/Vento/prepared/group_split_v2_train50k_test10k_nested5k/`
- benchmark root (`10k`):
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/Vento/benchmark/group_split_v2_train50k_test10k/all_methods_v1/`
- benchmark root (`5k` nested):
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/Vento/benchmark/group_split_v2_train50k_test5k_nested/all_methods_v1/`
- preparation resource summary:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/Vento/prepared/group_split_v2_train50k_test10k_nested5k/preparation_resource_summary.json`
- split summary:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/Vento/prepared/group_split_v2_train50k_test10k_nested5k/split_summary.json`
- benchmark status (`10k`):
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/Vento/benchmark/group_split_v2_train50k_test10k/all_methods_v1/scaleout_status.json`
- benchmark status (`5k` nested):
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/Vento/benchmark/group_split_v2_train50k_test5k_nested/all_methods_v1/scaleout_status.json`

Recorded process:

- Vento used the round-locked reduced ceiling (`50k` build with `10k` heldout)
- `10k` benchmark completed with `7/7` methods successful
- nested `5k` benchmark also completed with `7/7` methods successful

## What is actually completed

The datasets that are currently backed by second-wave preparation and second-wave
`10k` benchmark evidence are:

- `PHMap_Lung_Full_v43_light`
- `DISCO_hPBMCs`
- `mTCA`
- `HLCA_Core`
- `Vento`

These five have:

- repo-tracked execution report(s)
- repo-tracked experiment record(s)
- prepared asset roots under `~/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/<dataset>/prepared/`
- benchmark output roots under `~/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/<dataset>/benchmark/`
- `scaleout_status.json` proving per-method execution status

## What still needs reporting or execution

Based on current recorded evidence:

- `cd4` is currently excluded until valid raw counts are available
- `cd8` is currently excluded until valid raw counts are available

## Record-keeping note

When someone later says that a dataset "has already been tested", that claim
should be cross-checked against both:

- a repo-tracked `execution_report_*.md` and `experiment_record_*.md`
- a concrete tmp output root with `split_summary.json` or `scaleout_status.json`

This file should be updated before older per-dataset narratives are reused as
the current source of truth.
