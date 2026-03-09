# Formal third-wave scaling protocol

Date: `2026-03-06`

This protocol defines the formal third-wave scaling benchmark. It converts the
high-level plan into an execution contract and record-keeping contract.

## Scope

Main-panel datasets:

- `HLCA_Core`
- `PHMap_Lung_Full_v43_light`
- `mTCA`
- `DISCO_hPBMCs`

Supplementary reduced-ceiling dataset:

- `Vento`

Excluded:

- `cd4`
- `cd8`

## Goal

This round answers three paper-facing questions:

1. how does reference build size affect mapping quality
2. how does query size affect runtime and resource cost under a fixed build
3. whether the full pipeline is auditable and reproducible under strict formal
   records

## Formal scaling axes

### Build scaling

Use this grid:

- `10k`
- `20k`
- `30k`
- `50k`
- `100k`
- `150k`
- `200k`
- `300k`

Use one dedicated fixed query:

- `build_eval_fixed_10k`

Interpretation:

- build scaling is not train-only bookkeeping
- every build tier must be evaluated on the same dedicated `build_eval_fixed_10k`
  subset for that dataset

### Predict scaling

Use this fixed-build rule:

- reuse the already-trained `100k` build artifact from build scaling
- if `100k` is infeasible for a dataset, reuse the dataset-specific maximum
  feasible build artifact and label the result as a ceiling exception
- `Vento` supplementary runs reuse `50k`

Use this predict grid:

- `1k`
- `3k`
- `5k`
- `8k`
- `10k`
- `15k`
- `20k`

Optional tail:

- `50k`, only if the heldout pool supports it

## Split contract

Use group-aware split only.

Locked split keys:

- `PHMap_Lung_Full_v43_light`: `sample`
- `HLCA_Core`: `donor_id`
- `mTCA`: `orig.ident`
- `DISCO_hPBMCs`: `sample`
- `Vento`: `orig.ident`

Per dataset, materialize one formal split with:

- one build pool large enough for the maximum feasible build tier
- one standalone `build_eval_fixed_10k` subset
- one independent predict-scaling pool

Locked rule:

- the build-scaling `10k` query and the predict-scaling `10k` query must be
  different files
- both must come from the heldout side of the split

Recommended construction:

- nested build subsets sampled from one maximum build pool
- nested predict subsets sampled from one dedicated predict-scaling pool

## Preprocessing contract

Formal preparation must do the following before benchmark execution:

1. read source `.h5ad`
2. validate matrix contract and counts-layer contract
3. canonicalize gene IDs
4. generate one split plan
5. materialize all locked subsets
6. attach preprocessing metadata
7. record resource usage during preparation

The current formal contract still requires:

- `layers["counts"]` for atlasmtl preprocessing and scANVI-style comparators
- explicit `counts_layer` declaration in manifests
- explicit species declaration

## Method-specific path labeling

### CellTypist

Formal benchmark records must keep an explicit backend label:

- `formal_native`
- `formal_with_compat_shim`
- `wrapped_logreg`

Headline formal tables must not silently mix `wrapped_logreg` with native
CellTypist results.

### Seurat anchor transfer

Formal records must keep the backend label:

- `MapQuery`
- `TransferData-only`

Formal execution restriction:

- the isolated CPU `seurat_anchor_transfer` track is limited to build scaling
  at `10k / 20k / 30k / 50k`
- do not continue to `100k+` build tiers for the CPU seurat track
- retain CPU seurat predict scaling only for `Vento`
  - `Vento` supplementary reuses the `build=50k` reference ceiling
  - non-`Vento` CPU seurat predict results are exploratory-only and must not
    enter the main formal comparison tables
- within this restricted `10k / 20k / 30k / 50k` CPU seurat track, do not apply
  the generic `60-minute` stop rule

Reason:

- exploratory `HLCA` execution showed that larger CPU seurat build tiers become
  disproportionately expensive relative to the main round objective
- `Vento` is the only retained dataset where CPU seurat predict scaling remains
  fair against the shared reference ceiling because the formal fixed build is
  already `50k`
- on the main-panel datasets, retaining CPU seurat predict scaling would mix a
  method-specific practical ceiling with the shared `100k` predict-scaling
  contract used by the other methods

## Locked formal defaults

### scANVI

- `scvi_max_epochs=25`
- `scanvi_max_epochs=25`
- `query_max_epochs=20`
- `n_latent=20`
- `batch_size=256`
- `datasplitter_num_workers=0`
- GPU-only

### AtlasMTL

Shared:

- `num_threads=8`
- `max_epochs=50`
- `val_fraction=0.1`
- `early_stopping_patience=5`
- `input_transform=binary`
- `reference_storage=external`

CPU:

- `learning_rate=3e-4`
- `hidden_sizes=[256,128]`
- `batch_size=128`

GPU:

- `learning_rate=1e-3`
- `hidden_sizes=[1024,512]`
- `batch_size=512`

## Runtime fairness contract

Use the thread policy from:

- `documents/protocols/third_wave_fairness_protocol.md`

Mandatory formal settings:

- `OMP_NUM_THREADS=8`
- `MKL_NUM_THREADS=8`
- `OPENBLAS_NUM_THREADS=8`
- `NUMEXPR_NUM_THREADS=8`

Formal runtime/resource tables must include:

- `fairness_policy`
- `thread_policy`
- `runtime_fairness_degraded`
- `effective_threads_observed`
- `device_used`
- `method_backend_path`

## GPU execution environment rule

Formal GPU benchmark runs must be started from a non-sandbox shell session.

Locked rule:

- do not treat sandbox or restricted-session CUDA failures as comparator or
  method failures
- do not launch formal GPU batches from a sandboxed execution context when
  `torch.cuda.is_available()` is known to be unreliable there
- launch formal GPU batches directly from a normal shell session and record that
  execution mode in the report when relevant

Current motivating example:

- `HLCA_Core` formal GPU main batch needed to be started from a direct shell
  session because sandboxed execution previously returned
  `torch.cuda.is_available() = false`

## Long-runtime stop rule

Formal execution uses one wall-time guardrail for single-method runs:

- if a single method remains active for more than `60 minutes`, stop that
  method run manually
- record the event as `manual_stop_long_runtime`
- preserve existing partial artifacts and logs
- classify the event separately from method crash/failure unless the logs show a
  real software error

Exception:

- this generic guardrail does not apply to the restricted CPU seurat formal
  track after that track was narrowed to `10k / 20k / 30k / 50k` only

The current motivating examples are:

- `HLCA_Core` CPU sanity, where `seurat_anchor_transfer` on
  `build100k -> eval10k` was stopped after a long-running `>60m` execution
  window
- subsequent formal-round policy tightening, where CPU seurat was reduced to
  `10k / 20k / 30k / 50k` only

## Required output locations

Round-level protocol and execution docs:

- `plan/2026-03-06_formal_third_wave_scaling_plan.md`
- `documents/experiments/2026-03-06_formal_third_wave_execution_template.md`
- `documents/experiments/2026-03-06_formal_third_wave_round_status.md`

Preparation dossier:

- `documents/experiments/2026-03-06_formal_third_wave_scaling/`

Per-dataset tmp output root:

- `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/<dataset>/`

Required preparation files:

- `prepared/formal_split_v1/split_plan.json`
- `prepared/formal_split_v1/split_summary.json`
- `prepared/formal_split_v1/preprocessing_summary.json`
- `prepared/formal_split_v1/preparation_resource_summary.json`
- `prepared/formal_split_v1/dataset_ceiling_summary.json`

## Final reporting contract

Round-level final report:

- `documents/experiments/2026-03-06_formal_third_wave_round_report.md`

The final report must distinguish:

- main-panel datasets
- supplementary reduced-ceiling datasets
- engineering failures
- method failures
- infrastructure interruptions
