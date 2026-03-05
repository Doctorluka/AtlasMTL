# Second-wave scale-out benchmark plan

Date: **2026-03-04**  
Scope: enlarge the current reference-heldout benchmark workflow from first-wave
smoke runs to a larger, more reproducible **scale-out reference benchmark**
across the full reference-data suite.

This plan is execution-oriented. It is intended to be checked off during actual
implementation and runs.

Non-goals for this round:

- external query validation
- formal multi-level headline benchmark
- paper-final comparator ranking

Primary goal for this round:

- stress the current reference-heldout pipeline at larger reference scale
- lock a more robust and reproducible execution protocol
- expose remaining engineering bottlenecks before the paper-grade round

---

## 0) Locked round objective

- [ ] All **reference datasets** enter the second-wave benchmark roster
- [ ] Each dataset runs at least one **large-reference** heldout benchmark
- [ ] The benchmark runner emits stable resource-monitoring outputs for every method
- [ ] Split generation is reproducible and documented per dataset
- [ ] Comparator outputs remain organized per reference dossier
- [ ] Results are interpreted as **workflow scale-out validation**, not final paper claims

---

## 1) Dataset roster and round target sizes

### 1.1 Full roster to include in this round

- [x] `PHMap_Lung_Full_v43_light`
- [x] `HLCA_Core`
- [x] `mTCA` (rerun required: first attempt used wrong `species=human`; corrected to `mouse`)
- [x] `DISCO_hPBMCs`
- [x] `cd4` (excluded from current round: blocked on `2026-03-04` re-check; no usable raw counts layer)
- [x] `cd8` (excluded from current round: blocked on `2026-03-04` re-check; no usable raw counts layer)
- [x] `Vento`

### 1.2 Locked training-size target for second wave

Default target:

- [ ] `reference build target = 100k`

Default heldout prediction targets:

- [ ] `predict target = 5k`
- [ ] `predict target = 10k`

Interpretation:

- `100k` is the main scale-out build size for datasets that can support it
- `5k` and `10k` are two heldout inference scales for runtime + stability checks

### 1.3 Dataset-specific ceilings

| Dataset | Preferred build target | Preferred predict targets | Rule |
| --- | ---: | ---: | --- |
| `PHMap_Lung_Full_v43_light` | `100k` | `5k`, `10k` | standard |
| `HLCA_Core` | `100k` | `5k`, `10k` | standard |
| `mTCA` | `100k` | `5k`, `10k` | standard |
| `DISCO_hPBMCs` | `100k` | `5k`, `10k` | standard |
| `cd4` | `100k` | `5k`, `10k` | standard |
| `cd8` | `100k` | `5k`, `10k` | standard |
| `Vento` | `50k` default, `full` optional deployment-only | `5k`, `10k` if feasible | reduced ceiling |

Acceptance:

- [ ] Every dataset has one recorded final build ceiling before run start
- [ ] Any deviation from `100k` is explicitly documented
- [ ] `Vento` is not silently forced into the common `100k` target

---

## 2) Split strategy for second wave

### 2.1 Split principle

Continue using **group-aware split**. The split unit remains the sample-like
group, not individual cells.

Locked split keys:

- `PHMap_Lung_Full_v43_light` → `sample`
- `HLCA_Core` → `donor_id`
- `mTCA` → `orig.ident`
- `DISCO_hPBMCs` → `sample`
- `cd4` → `sample`
- `cd8` → `sample`
- `Vento` → `orig.ident`

### 2.2 Recommended second-wave pool design

For each reference dataset:

1. **group-isolated build pool**
2. **group-isolated heldout pool**
3. optional **reserve pool** only if naturally left over by the split

This round should still avoid a separate external validation pool. Internal
validation remains inside model training config.

### 2.3 Recommended second-wave target construction

For datasets with enough cells:

- [ ] build pool must support at least `100k`
- [ ] heldout pool must support at least `10k`
- [ ] a nested `5k` heldout subset is sampled from the same heldout pool

Recommended structure:

- `reference_train_100k.h5ad`
- `heldout_test_10k.h5ad`
- `heldout_test_5k.h5ad`

For `Vento`:

- [ ] preferred build target: `50k`
- [ ] preferred heldout target: `10k`
- [ ] if `10k` is not feasible after group split, record the largest feasible heldout size
- [ ] if project owner later wants a full-data deployment model, handle that separately outside benchmark

### 2.4 Candidate generation rule

Second wave should still use the existing **A+ split** logic:

- random candidate group splits
- rejection of invalid candidates
- deterministic selection of the best candidate

But the scoring objective should be tightened.

Priority order:

1. zero group leakage
2. build-size feasibility
3. heldout-size feasibility
4. target-label coverage in heldout
5. minimum per-label support in heldout
6. target-label coverage in build
7. overshoot minimization

### 2.5 Minimum label-support rule

This round should be stricter than first-wave smoke.

Recommended policy:

- hard fail if heldout sampled set has `< 2` target labels
- hard fail if build sampled set has `< 2` target labels
- hard fail if any target size cannot be reached
- warning if a heldout label has `< 10` cells
- warning if a build label has `< 25` cells

For very long-tail datasets:

- [ ] record low-support labels explicitly in split summary
- [ ] do not silently drop labels unless a protocolized exclusion rule is introduced

### 2.6 My recommendation for `Vento`

`Vento` should use a **reduced-ceiling heldout benchmark**:

- default split target: `50k build / 10k heldout`
- fallback split target: `50k build / max feasible heldout`

Reason:

- the atlas is too small to make `100k build` meaningful
- using nearly the full dataset for training would collapse the heldout test into
  a weak engineering artifact

Acceptance:

- [ ] Each dataset has one materialized split plan JSON
- [ ] Each dataset has one split summary JSON
- [ ] `5k` and `10k` heldout subsets are nested and reproducible when feasible

---

## 3) Comparator scope for second wave

Comparator set remains:

- [ ] `atlasmtl`
- [ ] `reference_knn`
- [ ] `celltypist`
- [ ] `scanvi`
- [ ] `singler`
- [ ] `symphony`
- [ ] `seurat_anchor_transfer`

Round purpose:

- check pipeline stability at larger scale
- check runtime/resource behavior
- expose scale-specific failures early

Not required in this round:

- per-method exhaustive hyperparameter optimization

Allowed engineering adjustment:

- comparator-specific stabilization that preserves a fair input contract

Examples:

- `celltypist` uses dedicated log1p-normalized comparator input copies
- `celltypist` formal trainer restoration is in progress, but the first
  small-scale validation did not complete successfully under the current local
  environment; second-wave benchmark interpretation should continue to treat
  active CellTypist runs as `wrapped_logreg`-based until formal validation is
  completed
- `seurat_anchor_transfer` may fall back from reference integration to
  `single_reference_pca`
- `seurat_anchor_transfer` may fall back from `MapQuery` to `TransferData`

Acceptance:

- [ ] every comparator method is attempted for every reference dataset
- [ ] every failure leaves `stdout.log`, `stderr.log`, and a summarized reason

---

## 4) Resource monitoring requirements

This round must explicitly strengthen the existing monitoring module.

### 4.1 Required fields per method

- [ ] train elapsed seconds
- [ ] predict elapsed seconds
- [ ] peak RSS
- [ ] average RSS when obtainable
- [ ] device used
- [ ] CPU core-equivalent usage when obtainable
- [ ] peak GPU memory when applicable
- [ ] throughput (`cells/s`)

### 4.2 Required fields per preprocessing step

For preparation scripts:

- [ ] counts detection time
- [ ] gene ID canonicalization time
- [ ] feature selection time
- [ ] split materialization time
- [ ] peak RSS during preparation if feasible

### 4.3 Reporting outputs

Each second-wave run should emit:

- [ ] `metrics.json`
- [ ] `summary.csv`
- [ ] `summary_by_domain.csv` when applicable
- [ ] resource summary table
- [ ] experiment record with monitoring notes

Acceptance:

- [ ] resource fields are no longer sparsely populated for only a subset of methods
- [ ] missing fields are explicitly reported as missing, not silently omitted

---

## 5) Execution order

### 5.1 Preparation stage

- [x] Materialize second-wave manifests for all reference datasets
- [ ] Materialize split plans
- [ ] Materialize prepared reference/heldout assets
- [x] Audit label support and split warnings before any comparator run

### 5.2 Pilot-to-scale order

Recommended order:

1. `DISCO_hPBMCs`
2. `PHMap_Lung_Full_v43_light`
3. `mTCA`
4. `HLCA_Core`
5. `Vento`

Reason:

- start with one stable engineering regression set (`DISCO`)
- then one hard atlas (`PH-Map`)
- then expand to the rest after the larger-scale path is trusted

### 5.3 Stop conditions

Pause scale-out and fix the protocol if:

- [ ] two or more comparators fail on the same dataset for contract reasons
- [ ] split generation produces severe label collapse
- [ ] resource monitoring outputs are missing or inconsistent

---

## 6) Repo-side record requirements

For every second-wave dataset dossier:

- [ ] updated protocol note
- [ ] execution checklist
- [ ] execution report
- [ ] experiment record
- [ ] explicit error-and-fix section

Recommended run naming:

- `<dataset_id>__ref100k__test10k__v1`
- `<dataset_id>__ref100k__test5k__v1`
- `Vento__ref50k__test10k__v1`

Acceptance:

- [ ] a later reader can reconstruct the run without shell-history access
- [ ] the rationale for every fallback is documented

---

## 7) Deliverables for the end of this round

- [x] full reference suite manifests for second wave
- [x] split plans for all active datasets in this round
- [x] prepared assets for all active datasets in this round
- [x] one large-scale smoke/benchmark pass per active dataset in this round
- [ ] updated resource-monitoring summary
- [ ] round summary report comparing stability across datasets
- [ ] list of blockers before the next paper-grade round

---

## 8) Current next action

- [x] write second-wave manifests and split protocol extensions
- [x] implement larger-scale split materialization (`100k`, `10k`, `5k`)
- [x] extend preparation logging with preprocessing resource accounting
- [x] start second-wave execution with `DISCO_hPBMCs`, then `PH-Map`
- [x] complete second-wave execution for `mTCA`, `HLCA_Core`, and `Vento`
- [x] mark `cd4` and `cd8` as excluded from this round due to raw-count contract blockers
