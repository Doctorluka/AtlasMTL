# Formal third-wave scaling plan

Date: **2026-03-06**  
Scope: lock the paper-facing formal benchmark design for third-wave scaling and
start with the dataset-preparation phase.

This round is no longer a smoke run or a parameter-search round. The goal is to
produce a stable, auditable execution contract for the formal benchmark.

## 0) Locked round scope

Main-panel datasets:

- `HLCA_Core`
- `PHMap_Lung_Full_v43_light`
- `mTCA`
- `DISCO_hPBMCs`

Supplementary reduced-ceiling dataset:

- `Vento`

Excluded in this round:

- `cd4`
- `cd8`

## 1) Locked method scope

The formal benchmark keeps the current runnable comparator set:

- `atlasmtl`
- `reference_knn`
- `celltypist`
- `scanvi`
- `singler`
- `symphony`
- `seurat_anchor_transfer`

Locked interpretation rules:

- `scanvi` is GPU-only in formal runs
- `atlasmtl_cpu` and `atlasmtl_gpu` are separate runtime/resource variants
- `celltypist` results must keep explicit backend-path labels and must not
  silently mix `wrapped_logreg` with formal native results in headline tables

## 2) Locked formal scaling design

### 2.1 Build scaling

Purpose:

- test whether larger reference build size changes final mapping quality
- record how build size changes training cost and end-to-end cost

Build grid:

- `10k`
- `20k`
- `30k`
- `50k`
- `100k`
- `150k`
- `200k`
- `300k`

Fixed query:

- one dedicated `build_eval_fixed_10k`

Locked rule:

- `build_eval_fixed_10k` must not be reused as the `10k` node in predict
  scaling

Ceiling rule:

- each dataset runs only the feasible subset of the grid after group-aware
  split
- `PHMap_Lung_Full_v43_light` may stop below `300k`
- `Vento` does not enter the main build-scaling panel

### 2.2 Predict scaling

Purpose:

- test how query size changes runtime, throughput, and stability under a fixed
  build artifact

Fixed build:

- reuse the `100k` artifact from build scaling
- if `100k` is infeasible for a dataset, reuse that dataset's maximum feasible
  build artifact and record it as a ceiling exception
- `Vento` supplementary runs reuse `50k`

Predict grid:

- `1k`
- `3k`
- `5k`
- `8k`
- `10k`
- `15k`
- `20k`

Optional tail:

- `50k`, only when the heldout pool and compute budget allow it

Locked rule:

- predict-scaling subsets must be nested within one dedicated predict-scaling
  pool
- the predict-scaling `10k` subset must differ from the build-scaling fixed
  `10k`

## 3) Locked split and preprocessing contract

Split policy:

- use group-aware splitting only
- fixed split keys:
  - `PHMap_Lung_Full_v43_light` -> `sample`
  - `HLCA_Core` -> `donor_id`
  - `mTCA` -> `orig.ident`
  - `DISCO_hPBMCs` -> `sample`
  - `Vento` -> `orig.ident`

Preparation must happen before formal benchmark execution:

1. audit the raw source matrix and metadata contract
2. verify or materialize `layers["counts"]`
3. canonicalize gene IDs
4. generate one group-aware split plan per dataset
5. materialize:
   - nested build subsets
   - one standalone `build_eval_fixed_10k`
   - one nested predict-scaling pool
6. derive preprocessing assets needed by the formal benchmark
7. record dataset ceilings and any omitted tiers

Locked dataset-preparation rule:

- all subsets must be fixed before any method benchmark begins
- all methods for one dataset must reuse the same prepared subset files

## 4) Locked defaults from pre-formal tuning

`scanvi`:

- `scvi_max_epochs=25`
- `scanvi_max_epochs=25`
- `query_max_epochs=20`
- `n_latent=20`
- `batch_size=256`
- `datasplitter_num_workers=0`
- GPU-only

`atlasmtl` shared:

- `num_threads=8`
- `max_epochs=50`
- `val_fraction=0.1`
- `early_stopping_patience=5`
- `input_transform=binary`
- `reference_storage=external`

`atlasmtl` CPU:

- `learning_rate=3e-4`
- `hidden_sizes=[256,128]`
- `batch_size=128`

`atlasmtl` GPU:

- `learning_rate=1e-3`
- `hidden_sizes=[1024,512]`
- `batch_size=512`

## 5) Required record locations

Round-level planning and protocol:

- `plan/2026-03-06_formal_third_wave_scaling_plan.md`
- `documents/protocols/formal_third_wave_scaling_protocol.md`
- `documents/protocols/experiment_protocol.md`
- `documents/protocols/third_wave_fairness_protocol.md`

Round-level operator docs:

- `documents/experiments/2026-03-06_formal_third_wave_execution_template.md`
- `documents/experiments/2026-03-06_formal_third_wave_round_status.md`

Round-level preparation dossier:

- `documents/experiments/2026-03-06_formal_third_wave_scaling/`

Per-dataset formal output root:

- `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/<dataset>/`

Required preparation outputs per dataset:

- `prepared/formal_split_v1/split_plan.json`
- `prepared/formal_split_v1/split_summary.json`
- `prepared/formal_split_v1/preprocessing_summary.json`
- `prepared/formal_split_v1/preparation_resource_summary.json`
- `prepared/formal_split_v1/dataset_ceiling_summary.json`
- prepared build/predict subset `.h5ad` files

Final round report destination:

- `documents/experiments/2026-03-06_formal_third_wave_round_report.md`

## 6) Execution order

First phase:

1. data audit
2. split generation
3. preprocessing and prepared subset materialization

Then:

4. manifest generation
5. build scaling
6. predict scaling
7. paper-table export and round summary

## 7) Runtime guardrail

Formal execution adds one stop rule for long-running single-method jobs:

- if one method run remains active for more than `60 minutes`, stop that method
  run and record it as `manual_stop_long_runtime`
- preserve all already-written partial artifacts
- do not treat this as a method crash unless logs show an actual failure
- record the stop reason, elapsed wall time, and blocked stage in the execution
  report

## 8) First-phase acceptance criteria

- each dataset has one auditable split plan
- each dataset has one ceiling summary
- build-scaling fixed `10k` and predict-scaling `10k` are non-identical
- predict-scaling subsets are nested
- `Vento` is marked supplementary in records
- resource summaries are emitted during preparation

## 9) Post-benchmark paper-phase checklist

This is the highest-priority follow-up framework after formal benchmark
execution reaches completion. The benchmark is not considered paper-ready until
the items below are closed.

### 9.1 Result freeze and execution closure

- freeze the final dataset roster, method roster, and retained-vs-exploratory
  result policy
- freeze the final comparator interpretation rules, especially:
  - `scanvi` GPU-only
  - `atlasmtl_cpu` vs `atlasmtl_gpu` split reporting
  - `seurat_anchor_transfer` supplementary / exploratory boundaries
- close every dataset dossier with:
  - execution record
  - error-fix note
  - retained result root
  - discarded or exploratory result root if applicable
- confirm no formal table depends on an in-progress or partially written point

### 9.2 Data and artifact integrity audit

- verify that every retained point has:
  - `scaleout_status.json`
  - `runtime_manifest.yaml`
  - method-level `summary.csv`
  - method-level `stdout.log`
  - method-level `stderr.log`
- verify that retained point counts match the intended manifests per dataset and
  track
- verify that prepared split assets, model artifacts, and result summaries are
  still resolvable from the paths recorded in the report
- record any missing or exploratory-only outputs explicitly instead of leaving
  silent gaps

### 9.3 Scientific result review

- perform a structured anomaly review for:
  - unexpected performance drops
  - unexpected resource spikes
  - fairness-degraded runs
  - method-specific outliers such as `Vento seurat`
- compare headline method behavior against method papers or known public
  benchmarks when possible
- decide for each anomalous result whether it will be:
  - retained with explanation
  - rerun
  - moved to supplementary
  - excluded from headline comparison

### 9.4 Formal table and figure production

- generate final paper-facing tables for:
  - performance
  - resource consumption
  - fairness / execution contract
  - dataset completion and ceiling summary
- generate final figures for:
  - build-scaling curves
  - predict-scaling curves
  - performance-resource tradeoff plots
  - optional supplementary method-specific plots
- require the report to name which tables and figures are main-text versus
  supplementary

### 9.5 Methods and results writing package

- write a benchmark methods section that exactly matches the executed protocol
- write a results narrative that distinguishes:
  - headline findings
  - efficiency findings
  - supplementary comparator findings
  - exploratory findings
- document why any retained method-specific exception exists, for example:
  - `Vento` supplementary reduced-ceiling status
  - `seurat_anchor_transfer` restricted formal role

### 9.6 Final rerun gate

- allow only targeted reruns after completion of the review above
- do not reopen broad parameter tuning or pipeline redesign inside the formal
  round
- any rerun must have:
  - a stated reason
  - a stated affected table or figure
  - an execution note appended to the round report

### 9.7 Final deliverables

The round is only fully closed when the following deliverables exist:

- one finalized round-level formal report
- one finalized round-level table index
- one finalized round-level figure index
- one finalized dataset-by-dataset execution summary
- one explicit main-text versus supplementary result map
- one reproducibility index pointing from report text to runtime artifact roots
