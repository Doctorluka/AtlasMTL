# Second-Wave Execution Template

This file is the operator-facing template for the current second-wave
reference-heldout benchmark round.

Use it for three tasks:

1. immediately locate the critical files for a completed dataset
2. self-check whether a dataset run is complete and paper-usable
3. start a new dataset run by copying a known-good completed pattern

Current completed reference examples:

- `PHMap_Lung_Full_v43_light`
- `DISCO_hPBMCs`
- `mTCA`
- `HLCA_Core`
- `Vento`

Current exclusions / pending items:

- `cd4`: excluded for now because no acceptable raw counts source is available
- `cd8`: excluded for now because no acceptable raw counts source is available

## Open These Files First

Always open these first:

- round status:
  `documents/experiments/2026-03-04_second_wave_round_status.md`
- execution protocol:
  `documents/protocols/reference_heldout_scaleout_execution.md`
- this operator template:
  `documents/experiments/2026-03-04_second_wave_execution_template.md`

If you are starting from a completed dataset, open the dataset dossier next:

- `documents/experiments/2026-03-03_phmap_benchmark/`
- `documents/experiments/2026-03-03_projectsvr_disco_benchmark/`
- `documents/experiments/2026-03-04_projectsvr_mtca_benchmark/`
- `documents/experiments/2026-03-03_hlca_benchmark/`
- `documents/experiments/2026-03-04_projectsvr_vento_benchmark/`

## What A Complete Dataset Must Contain

For a dataset to count as completed in this round, it must have all of the
following:

### 1) Repo-side manifests and scripts

- one prep manifest
- one runtime manifest for `10k`
- one runtime manifest for `5k`
- one dataset prep wrapper script

### 2) Tmp prepared outputs

Under:

- `~/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/<dataset>/prepared/<split_name>/`

Must exist:

- `reference_train.h5ad`
- `heldout_test_10k.h5ad`
- `heldout_test_5k.h5ad`
- `feature_panel.json`
- `split_plan.json`
- `split_summary.json`
- `preprocessing_summary.json`
- `preparation_resource_summary.json`

### 3) Tmp benchmark outputs

Under:

- `~/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/<dataset>/benchmark/<split_name>/<run_name>/`

Must exist:

- `scaleout_status.json`
- `runtime_manifest.yaml`
- `runtime_manifest_celltypist.yaml`
- `runs/<method>/summary.csv`
- `runs/<method>/metrics.json`

### 4) Repo-side markdown records

Must exist in the owning dossier:

- `results_summary/execution_report_<date>_scaleout_preparation.md` or an
  equivalent preparation report
- `results_summary/experiment_record_<date>_scaleout_preparation.md` or an
  equivalent preparation record
- `results_summary/execution_report_<date>_scaleout_benchmark_10k.md`
- `results_summary/experiment_record_<date>_scaleout_benchmark_10k.md`

## The Most Important Files

If you only inspect a small number of files, inspect these first.

### A) Split and preprocessing truth

- `split_summary.json`
- `preprocessing_summary.json`
- `preparation_resource_summary.json`

These tell you:

- what split was actually materialized
- whether counts and gene-ID contracts were satisfied
- how expensive preparation was

### B) Benchmark truth

- `scaleout_status.json`
- `runs/<method>/metrics.json`
- `runs/<method>/summary.csv`

These tell you:

- whether every method really ran
- exact per-method metrics
- exact runtime/resource fields used later for tables

### C) Human-readable narrative

- `execution_report_*.md`
- `experiment_record_*.md`

These tell you:

- what happened
- what failed
- what was corrected
- what caveats still remain

For paper-grade traceability, the machine-truth files are the JSON/CSV files
above; the markdown files are the audit trail that explains them.

## Completed Dataset Map

Use these as the current known-good templates.

### 1) `PHMap_Lung_Full_v43_light`

Repo-side anchors:

- prep manifest:
  `documents/experiments/2026-03-03_phmap_benchmark/manifests/reference_heldout/PHMap_Lung_Full_v43_light__anno_lv4__scaleout_prep_v1.yaml`
- runtime manifest `10k`:
  `documents/experiments/2026-03-03_phmap_benchmark/manifests/reference_heldout/PHMap_Lung_Full_v43_light__anno_lv4__scaleout_runtime_10k_v1.yaml`
- runtime manifest `5k`:
  `documents/experiments/2026-03-03_phmap_benchmark/manifests/reference_heldout/PHMap_Lung_Full_v43_light__anno_lv4__scaleout_runtime_5k_v1.yaml`
- prep script:
  `documents/experiments/2026-03-03_phmap_benchmark/scripts/run_prepare_scaleout.sh`
- benchmark report:
  `documents/experiments/2026-03-03_phmap_benchmark/results_summary/execution_report_2026-03-04_scaleout_benchmark_10k.md`
- benchmark record:
  `documents/experiments/2026-03-03_phmap_benchmark/results_summary/experiment_record_2026-03-04_scaleout_benchmark_10k.md`

Tmp anchors:

- prepared root:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/PHMap_Lung_Full_v43_light/prepared/group_split_v2_train100k_test10k_nested5k/`
- benchmark root:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/PHMap_Lung_Full_v43_light/benchmark/group_split_v2_train100k_test10k/all_methods_v1/`
- critical files:
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/PHMap_Lung_Full_v43_light/prepared/group_split_v2_train100k_test10k_nested5k/split_summary.json`
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/PHMap_Lung_Full_v43_light/prepared/group_split_v2_train100k_test10k_nested5k/preprocessing_summary.json`
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/PHMap_Lung_Full_v43_light/prepared/group_split_v2_train100k_test10k_nested5k/preparation_resource_summary.json`
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/PHMap_Lung_Full_v43_light/benchmark/group_split_v2_train100k_test10k/all_methods_v1/scaleout_status.json`

### 2) `DISCO_hPBMCs`

Repo-side anchors:

- prep manifest:
  `documents/experiments/2026-03-03_projectsvr_disco_benchmark/manifests/reference_heldout/DISCO_hPBMCs__cell_subtype__scaleout_prep_v1.yaml`
- runtime manifest `10k`:
  `documents/experiments/2026-03-03_projectsvr_disco_benchmark/manifests/reference_heldout/DISCO_hPBMCs__cell_subtype__scaleout_runtime_10k_v1.yaml`
- runtime manifest `5k`:
  `documents/experiments/2026-03-03_projectsvr_disco_benchmark/manifests/reference_heldout/DISCO_hPBMCs__cell_subtype__scaleout_runtime_5k_v1.yaml`
- prep script:
  `documents/experiments/2026-03-03_projectsvr_disco_benchmark/scripts/run_prepare_scaleout.sh`
- benchmark report:
  `documents/experiments/2026-03-03_projectsvr_disco_benchmark/results_summary/execution_report_2026-03-04_scaleout_benchmark_10k.md`
- benchmark record:
  `documents/experiments/2026-03-03_projectsvr_disco_benchmark/results_summary/experiment_record_2026-03-04_scaleout_benchmark_10k.md`

Tmp anchors:

- prepared root:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/DISCO_hPBMCs/prepared/group_split_v2_train100k_test10k_nested5k/`
- benchmark root:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/DISCO_hPBMCs/benchmark/group_split_v2_train100k_test10k/all_methods_v2/`
- critical files:
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/DISCO_hPBMCs/prepared/group_split_v2_train100k_test10k_nested5k/split_summary.json`
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/DISCO_hPBMCs/prepared/group_split_v2_train100k_test10k_nested5k/preprocessing_summary.json`
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/DISCO_hPBMCs/prepared/group_split_v2_train100k_test10k_nested5k/preparation_resource_summary.json`
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/DISCO_hPBMCs/benchmark/group_split_v2_train100k_test10k/all_methods_v2/scaleout_status.json`

### 3) `mTCA`

Repo-side anchors:

- prep manifest:
  `documents/experiments/2026-03-04_projectsvr_mtca_benchmark/manifests/reference_heldout/mTCA__Cell_type_level3__scaleout_prep_v1.yaml`
- runtime manifest `10k`:
  `documents/experiments/2026-03-04_projectsvr_mtca_benchmark/manifests/reference_heldout/mTCA__Cell_type_level3__scaleout_runtime_10k_v1.yaml`
- runtime manifest `5k`:
  `documents/experiments/2026-03-04_projectsvr_mtca_benchmark/manifests/reference_heldout/mTCA__Cell_type_level3__scaleout_runtime_5k_v1.yaml`
- prep script:
  `documents/experiments/2026-03-04_projectsvr_mtca_benchmark/scripts/run_prepare_scaleout.sh`
- benchmark report:
  `documents/experiments/2026-03-04_projectsvr_mtca_benchmark/results_summary/execution_report_2026-03-04_scaleout_benchmark_10k.md`
- benchmark record:
  `documents/experiments/2026-03-04_projectsvr_mtca_benchmark/results_summary/experiment_record_2026-03-04_scaleout_benchmark_10k.md`

Tmp anchors:

- prepared root:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/mTCA/prepared/group_split_v2_train100k_test10k_nested5k/`
- invalid first benchmark root:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/mTCA/benchmark/group_split_v2_train100k_test10k/all_methods_v1/`
- corrected benchmark root:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/mTCA/benchmark/group_split_v2_train100k_test10k/all_methods_v2_mouse_fix/`
- critical files:
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/mTCA/prepared/group_split_v2_train100k_test10k_nested5k/split_summary.json`
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/mTCA/prepared/group_split_v2_train100k_test10k_nested5k/preprocessing_summary.json`
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/mTCA/prepared/group_split_v2_train100k_test10k_nested5k/preparation_resource_summary.json`
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/mTCA/benchmark/group_split_v2_train100k_test10k/all_methods_v2_mouse_fix/scaleout_status.json`

Special correction record:

- the invalid first run must not be reused
- the authoritative corrected run is:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/mTCA/benchmark/group_split_v2_train100k_test10k/all_methods_v2_mouse_fix/`

### 4) `HLCA_Core`

Repo-side anchors:

- prep manifest:
  `documents/experiments/2026-03-03_hlca_benchmark/manifests/reference_heldout/HLCA_Core__ann_level_5__scaleout_prep_v1.yaml`
- runtime manifest `10k`:
  `documents/experiments/2026-03-03_hlca_benchmark/manifests/reference_heldout/HLCA_Core__ann_level_5__scaleout_runtime_10k_v1.yaml`
- runtime manifest `5k`:
  `documents/experiments/2026-03-03_hlca_benchmark/manifests/reference_heldout/HLCA_Core__ann_level_5__scaleout_runtime_5k_v1.yaml`
- prep script:
  `documents/experiments/2026-03-03_hlca_benchmark/scripts/run_prepare_scaleout.sh`
- benchmark report:
  `documents/experiments/2026-03-03_hlca_benchmark/results_summary/execution_report_2026-03-05_scaleout_benchmark_10k.md`
- benchmark record:
  `documents/experiments/2026-03-03_hlca_benchmark/results_summary/experiment_record_2026-03-05_scaleout_benchmark_10k.md`

Tmp anchors:

- prepared root:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/prepared/group_split_v2_train100k_test10k_nested5k/`
- benchmark root:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/benchmark/group_split_v2_train100k_test10k/all_methods_v1/`
- critical files:
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/prepared/group_split_v2_train100k_test10k_nested5k/split_summary.json`
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/prepared/group_split_v2_train100k_test10k_nested5k/preprocessing_summary.json`
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/prepared/group_split_v2_train100k_test10k_nested5k/preparation_resource_summary.json`
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/benchmark/group_split_v2_train100k_test10k/all_methods_v1/scaleout_status.json`

### 5) `Vento`

Repo-side anchors:

- prep manifest:
  `documents/experiments/2026-03-04_projectsvr_vento_benchmark/manifests/reference_heldout/Vento__annotation__scaleout_prep_v1.yaml`
- runtime manifest `10k`:
  `documents/experiments/2026-03-04_projectsvr_vento_benchmark/manifests/reference_heldout/Vento__annotation__scaleout_runtime_10k_v1.yaml`
- runtime manifest `5k`:
  `documents/experiments/2026-03-04_projectsvr_vento_benchmark/manifests/reference_heldout/Vento__annotation__scaleout_runtime_5k_v1.yaml`
- prep script:
  `documents/experiments/2026-03-04_projectsvr_vento_benchmark/scripts/run_prepare_scaleout.sh`
- benchmark report (`10k`):
  `documents/experiments/2026-03-04_projectsvr_vento_benchmark/results_summary/execution_report_2026-03-04_scaleout_benchmark_10k.md`
- benchmark record (`10k`):
  `documents/experiments/2026-03-04_projectsvr_vento_benchmark/results_summary/experiment_record_2026-03-04_scaleout_benchmark_10k.md`
- benchmark report (`5k` nested):
  `documents/experiments/2026-03-04_projectsvr_vento_benchmark/results_summary/execution_report_2026-03-04_scaleout_benchmark_5k.md`
- benchmark record (`5k` nested):
  `documents/experiments/2026-03-04_projectsvr_vento_benchmark/results_summary/experiment_record_2026-03-04_scaleout_benchmark_5k.md`

Tmp anchors:

- prepared root:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/Vento/prepared/group_split_v2_train50k_test10k_nested5k/`
- benchmark root (`10k`):
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/Vento/benchmark/group_split_v2_train50k_test10k/all_methods_v1/`
- benchmark root (`5k` nested):
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/Vento/benchmark/group_split_v2_train50k_test5k_nested/all_methods_v1/`
- critical files:
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/Vento/prepared/group_split_v2_train50k_test10k_nested5k/split_summary.json`
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/Vento/prepared/group_split_v2_train50k_test10k_nested5k/preprocessing_summary.json`
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/Vento/prepared/group_split_v2_train50k_test10k_nested5k/preparation_resource_summary.json`
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/Vento/benchmark/group_split_v2_train50k_test10k/all_methods_v1/scaleout_status.json`
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/Vento/benchmark/group_split_v2_train50k_test5k_nested/all_methods_v1/scaleout_status.json`

## Self-Check Template

Before you call a dataset complete, check these in order.

### Step 1: Dossier exists

Check:

- dossier root exists
- `manifests/reference_heldout/` exists
- `scripts/` exists
- `results_summary/` exists

### Step 2: Prep manifests are consistent

Check:

- prep manifest points to the correct source dataset contract
- runtime `10k` and `5k` manifests point to the same prepared root
- split key is correct for the dataset
- target label is correct for the dataset
- species is correct
- counts-layer contract is explicit

### Step 3: Prepared outputs are complete

Check these files exist under the prepared root:

- `reference_train.h5ad`
- `heldout_test_10k.h5ad`
- `heldout_test_5k.h5ad`
- `feature_panel.json`
- `split_plan.json`
- `split_summary.json`
- `preprocessing_summary.json`
- `preparation_resource_summary.json`

Then inspect:

- `split_summary.json`
  - no group leakage
  - requested sizes are achieved
  - low-support label warnings are explicit
- `preprocessing_summary.json`
  - counts source is explicit
  - species is correct
  - feature panel size is correct
- `preparation_resource_summary.json`
  - elapsed / RSS / CPU-equivalent fields are present

### Step 4: Benchmark outputs are complete

Check:

- `scaleout_status.json` exists
- every required method has `status: success`
- `runtime_manifest.yaml` exists
- `runtime_manifest_celltypist.yaml` exists
- every method directory contains `summary.csv` and `metrics.json`

### Step 5: Markdown audit trail is complete

Check:

- preparation execution report exists
- preparation experiment record exists
- benchmark execution report exists
- benchmark experiment record exists
- failure/fix history is recorded when relevant

### Step 6: Paper-grade minimum record is present

The minimum set to preserve for future paper work is:

- prep manifest
- runtime manifest `10k`
- runtime manifest `5k`
- `split_summary.json`
- `preprocessing_summary.json`
- `preparation_resource_summary.json`
- `scaleout_status.json`
- every `runs/<method>/metrics.json`
- every `runs/<method>/summary.csv`
- benchmark execution report
- benchmark experiment record

## How To Start A New Dataset From A Completed Template

Use one of the completed datasets as a structural template.

### Option A: use `PHMap` as the default template

Use when:

- counts are already in `layers["counts"]`
- label transfer is harder and closer to the main benchmark target

Copy these patterns:

- dossier structure from:
  `documents/experiments/2026-03-03_phmap_benchmark/`
- prep wrapper from:
  `documents/experiments/2026-03-03_phmap_benchmark/scripts/run_prepare_scaleout.sh`
- report layout from:
  `documents/experiments/2026-03-03_phmap_benchmark/results_summary/`

### Option B: use `DISCO` as the comparator-wrapper template

Use when:

- you need the cleanest example of the scale-out wrapper around `celltypist`
- you want the easiest current reference example

Copy these patterns:

- dossier structure from:
  `documents/experiments/2026-03-03_projectsvr_disco_benchmark/`
- prep wrapper from:
  `documents/experiments/2026-03-03_projectsvr_disco_benchmark/scripts/run_prepare_scaleout.sh`
- benchmark wrapper behavior from:
  `documents/experiments/common/run_reference_heldout_scaleout_benchmark.py`

### Option C: use `mTCA` as the failure-recovery template

Use when:

- species / gene-ID / contract mistakes are plausible
- you want the best example of how to record an invalid first attempt and a
  corrected rerun

Copy these patterns:

- dossier structure from:
  `documents/experiments/2026-03-04_projectsvr_mtca_benchmark/`
- prep wrapper from:
  `documents/experiments/2026-03-04_projectsvr_mtca_benchmark/scripts/run_prepare_scaleout.sh`
- failure/fix reporting style from:
  `documents/experiments/2026-03-04_projectsvr_mtca_benchmark/results_summary/experiment_record_2026-03-04_scaleout_benchmark_10k.md`

## New-Dataset Launch Checklist

When opening a new dataset, do this in order.

1. Create or select the dataset dossier under `documents/experiments/`.
2. Copy a completed dataset's three manifest files and rename them.
3. Edit only:
   - source dataset path
   - dataset name
   - target label
   - split key
   - domain key
   - species
   - build ceiling
   - heldout sizes if the dataset needs a reduced ceiling
4. Copy and edit the dataset `run_prepare_scaleout.sh`.
5. Run preparation first.
6. Do not start benchmark until the self-check above passes for the prepared root.
7. Launch the benchmark with the dataset runtime `10k` manifest.
8. Immediately write:
   - preparation execution report
   - preparation experiment record
   - benchmark execution report
   - benchmark experiment record
9. If the first run is invalid, keep the invalid path recorded and create a new
   corrected run directory instead of overwriting history.

## Commands To Reuse

Preparation is launched from the dataset-specific wrapper script, for example:

- `documents/experiments/2026-03-03_phmap_benchmark/scripts/run_prepare_scaleout.sh`
- `documents/experiments/2026-03-03_projectsvr_disco_benchmark/scripts/run_prepare_scaleout.sh`
- `documents/experiments/2026-03-04_projectsvr_mtca_benchmark/scripts/run_prepare_scaleout.sh`
- `documents/experiments/2026-03-03_hlca_benchmark/scripts/run_prepare_scaleout.sh`

Benchmark execution is launched through:

- `documents/experiments/common/run_reference_heldout_scaleout_benchmark.py`

The canonical completed benchmark command pattern is:

```bash
/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python \
  documents/experiments/common/run_reference_heldout_scaleout_benchmark.py \
  --dataset-manifest <runtime_manifest_10k.yaml> \
  --output-dir <tmp benchmark output root> \
  --methods atlasmtl reference_knn celltypist scanvi singler symphony seurat_anchor_transfer \
  --device cpu
```

## If Something Is Missing

Use this rule:

- if repo markdown exists but tmp outputs do not, the run is not yet
  recoverably complete
- if tmp outputs exist but repo markdown does not, write the markdown before
  calling the dataset complete
- if both are missing, treat the dataset as not run
- if the run exists but the contract is wrong, keep the invalid run recorded and
  rerun into a new output directory

For the current round, pending benchmark execution items are:

- none in the active roster (`PHMap`, `DISCO`, `mTCA`, `HLCA`, `Vento`)
