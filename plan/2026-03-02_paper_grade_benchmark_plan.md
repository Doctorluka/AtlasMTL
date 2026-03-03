# Paper-grade benchmark plan (expert checklist aligned)

Date: **2026-03-02**  
Repo root: `/home/data/fhz/project/phmap_package/atlasmtl`  
Source checklist: `documents/reference/experts/experts_checklist.md`  
Motivation: `documents/reference/experts/experts_suggesstions.md`

This plan converts the expert reviewer checklist into a **trackable execution
plan** with explicit deliverables and acceptance criteria.

Non-goal: performing dataset cleaning inside this plan. Dataset cleaning is
handled externally and delivered back as “cleaned AnnData assets + notes”.

---

## 0) Progress dashboard (fill as you go)

- **Owner (data cleaning)**: _TBD_ (external)  
- **Owner (code/protocol/benchmark outputs)**: _TBD_  
- **Target tier**: methods top-tier (Nat Methods/Nat Biotech style evidence chain)

**Current status**

- [ ] Tier-0 outputs (metrics/plots/tables) are standardized and always emitted
- [ ] Dataset suite has ≥2–3 pairs + ≥1 shift scenario wired into manifests
- [ ] Reference-heldout benchmark split protocol is fixed
- [ ] External query validation is explicitly separated from accuracy benchmark
- [ ] Dataset-specific feasible training ceilings are recorded before runs start
- [ ] Threshold selection is protocolized (no per-run hand tuning)
- [ ] Resource + scaling outputs exist (CPU-first)
- [ ] Comparator protocol fairness is documented and exported per run

**Locked first-wave flow-through pilots**

- [x] `PH-Map`: `5k` reference build + `1k` prediction, standard `layers["counts"]` path
- [x] `DISCO_hPBMCs`: `5k` reference build + `1k` prediction, `adata.X -> layers["counts"]` validation path
- [x] `domain_key` for both first-wave pilots: `sample`
- [x] counts detection is the first required preprocessing gate
- [x] gene ID canonicalization runs after counts validation and before split materialization
- [x] all supported comparators are included in the first flow-through round
- [x] runtime assets live under `~/tmp/`, while repo-side dossiers retain reports, summaries, and discussion notes

**Run log (append-only)**

| Date | Run ID / output dir | Dataset | Scenario | Seeds | Notes |
| --- | --- | --- | --- | ---: | --- |
|  |  |  |  |  |  |

---

## 1) “Paper-grade” required artifacts (must exist for every formal run)

Reference protocol: `documents/protocols/experiment_protocol.md`

### 1.1 Required machine-readable output

- [ ] `metrics.json` includes, per method and per level:
  - [ ] accuracy / macro-F1 / balanced accuracy
  - [ ] coverage / reject_rate / covered_accuracy / risk
  - [ ] calibration: ECE (+ Brier recommended)
  - [ ] selective prediction: AURC
- [ ] `metrics.json` includes, when multi-level labels are present:
  - [ ] hierarchy: edge path consistency + full-path accuracy/coverage/covered accuracy
  - [ ] **cross-parent error rate** (lv4 predicted parent != true parent) when rules exist
- [ ] `metrics.json` includes runtime/resource usage:
  - [ ] train/predict elapsed seconds
  - [ ] train/predict items_per_second (cells/s)
  - [ ] train/predict peak RSS (GB)
  - [ ] GPU avg/peak memory (when applicable)
- [ ] `metrics.json` includes per-method `input_contract` (matrix sources, counts layer, normalization mode, feature alignment)

### 1.2 Required paper-ready plots (exported per run)

- [ ] reliability diagram (lv4) **before/after calibration** (when calibration is available)
- [ ] risk–coverage curve (lv4) with AURC annotation
- [ ] resource scaling:
  - [ ] cells vs time
  - [ ] cells vs peak RSS
  - [ ] cells vs throughput (cells/s)
- [ ] confusion/hierarchy diagnostic:
  - [ ] cross-parent error rate summary plot (or table rendered as figure)

### 1.3 Required paper tables (exported per run)

Use existing exporters:

- `benchmark/reports/generate_markdown_report.py`
- `benchmark/reports/export_paper_tables.py`

Checklist:

- [ ] `paper_tables/main_comparison.csv|.md` includes: ECE/Brier/AURC columns
- [ ] `paper_tables/comparator_protocol.csv|.md` exists (fairness + matrix semantics)
- [ ] `paper_tables/runtime_resources.csv|.md` exists (time/RSS/VRAM/throughput)
- [ ] If multi-level AtlasMTL run:
  - [ ] `paper_tables/atlasmtl_analysis.csv|.md` contains Unknown + KNN behavior + hierarchy metrics

---

## 2) Dataset suite plan (reference/query facts filled, pairing rules partially pending)

Normalized reference-data inventory:

- `data_registry/reference_data_inventory_2026-03-03.md`
- `data_registry/query_data_inventory_2026-03-03.md`

The plan below records **confirmed reference/query-side facts** and keeps only
the still-unresolved pairing/policy/shift decisions as placeholders.

### 2.1 Dataset: PH-Map full lung atlas

- **Dataset ID**: `PHMap_Lung_Full_v43_light`
- **Confirmed reference path**: `/home/data/fhz/project/phmap_package/data/real_test/phmap/phmap_clean.h5ad`

**Confirmed reference-side contract**

- [x] Cleaned reference `.h5ad` path recorded
- [x] Reference labels available for hierarchical training: `anno_lv1`, `anno_lv2`, `anno_lv3`, `anno_lv4`
- [x] Single-level coarse benchmark label: `anno_lv2`
- [x] Single-level fine benchmark label: `anno_lv4`
- [x] `var_names_type`: symbol-like
- [x] Counts contract: `layers["counts"]`
- [x] Geometry assets available: `obsm["X_scANVI"]`, `obsm["X_umap"]`, `obsp["connectivities"]`, `obsp["distances"]`

**Confirmed query-side facts**

- [x] Cleaned query `.h5ad` path recorded: `/home/data/fhz/project/phmap_package/data/real_test/query_data/query_PH.h5ad`
- [x] Query has `layers["counts"]`
- [x] Query `var_names` are Symbol-like with Ensembl-like IDs in `var["gene_ids"]`
- [x] Query is intended for external validation; final correctness review is marker-based unless formal query labels are later approved

**Pending pairing / protocol fields**

- [ ] Canonicalization notes (mapping resource + policies): `TBD`
- [x] Reference sample-like grouping field observed: `sample`
- [ ] Domain key for grouped reporting: `TBD` (candidates: `study`, `dataset`, `sample`, `batch`, `group`)
- [ ] Final decision: whether this pair is a formal label-based benchmark or a marker-validated external transfer case

**Planned scenarios**

- [x] Primary external-validation pair: `PH-Map ref -> query_PH`
- [ ] In-domain baseline eligibility confirmed
- [ ] Scaling points (subsample query to multiple cell counts; protocolized)

### 2.2 Dataset: HLCA core

- **Dataset ID**: `HLCA_Core`
- **Confirmed reference path**: `/home/data/fhz/project/phmap_package/data/real_test/HLCA/hlca_clean.h5ad`

**Confirmed reference-side contract**

- [x] Cleaned reference `.h5ad` path recorded
- [x] Reference labels available for hierarchical training: `ann_level_1`, `ann_level_2`, `ann_level_3`, `ann_level_4`, `ann_level_5`
- [x] Single-level coarse benchmark label: `ann_level_3`
- [x] Single-level fine benchmark label: `ann_level_5`
- [x] `ann_finest_level` is not part of the cleaned runtime contract
- [x] `var_names_type`: Ensembl-like
- [x] Counts contract: `layers["counts"]`
- [x] `adata.X` semantics: non-negative log-normalized expression
- [x] Geometry assets available: `obsm["X_scANVI"]`, `obsm["X_umap"]`, `obsp["connectivities"]`, `obsp["distances"]`

**Confirmed query-side facts**

- [x] Cleaned query `.h5ad` path recorded: `/home/data/fhz/project/phmap_package/data/real_test/query_data/hlca_query_GSE302339.h5ad`
- [x] Query has `layers["counts"]`
- [x] Query `adata.X` is non-negative log-normalized expression
- [x] Query `var_names` are mixed Symbol/Ensembl; canonical IDs live in `var["ENSEMBL"]`
- [x] Query provides `obs["Gold"]`, but use as formal benchmark truth still requires project-owner approval

**Pending pairing / protocol fields**

- [x] Reference sample-like grouping field observed: `donor_id`
- [ ] Domain key for grouped reporting: `TBD` (likely: `donor_id`, `GSE_id`, `Subject group`)
- [ ] Final decision: whether `Gold` is accepted as evaluation truth or kept marker-only
- [ ] Final gene canonicalization rule for mixed `var_names`

**Planned scenarios**

- [x] Primary external-validation pair: `HLCA ref -> hlca_query_GSE302339`
- [ ] In-domain baseline eligibility confirmed
- [ ] Scaling points (subsample query to multiple cell counts; protocolized)

### 2.3 Dataset: ProjectSVR package

- **Dataset ID**: `ProjectSVR_Package`
- **Confirmed reference paths**:
  - `mTCA`: `/home/data/fhz/project/phmap_package/data/real_test/ProjectSVR/reference_atlas/mTCA.h5ad`
  - `DISCO_hPBMCs`: `/home/data/fhz/project/phmap_package/data/real_test/ProjectSVR/reference_atlas/DISCO_hPBMCs.h5ad`
  - `cd4`: `/home/data/fhz/project/phmap_package/data/real_test/ProjectSVR/reference_atlas/cd4.h5ad`
  - `cd8`: `/home/data/fhz/project/phmap_package/data/real_test/ProjectSVR/reference_atlas/cd8.h5ad`
  - `vento`: `/home/data/fhz/project/phmap_package/data/real_test/ProjectSVR/reference_atlas/vento.h5ad`

**Confirmed reference-side contract**

- [x] `mTCA` hierarchical training labels: `Cell_type_level1`, `Cell_type_level2`, `Cell_type_level3`
- [x] `mTCA` coarse/fine benchmark labels: `Cell_type_level2` / `Cell_type_level3`
- [x] `DISCO_hPBMCs` is not for hierarchical training
- [x] `DISCO_hPBMCs` coarse/fine benchmark labels: `cell_type` / `cell_subtype`
- [x] `cd4` is not for hierarchical training
- [x] `cd4` coarse/fine benchmark labels: `cell_type` / `cell_subtype`
- [x] `cd8` is not for hierarchical training
- [x] `cd8` coarse/fine benchmark labels: `cell_type` / `cell_subtype`
- [x] `vento` is single-label only: `annotation`
- [x] `mTCA`, `DISCO_hPBMCs`, `cd4`, `cd8`, `vento` all have count-like integer `adata.X`
- [x] Absence of `layers["counts"]` is intentional for auto-handling validation in current atlasmtl flow

**Confirmed query-side facts**

- [x] `DISCO_hPBMCs` external query path recorded: `/home/data/fhz/project/phmap_package/data/real_test/ProjectSVR/query_data/pbmc_query.h5ad`
- [x] `pbmc_query` has count-like integer `adata.X` and no `layers["counts"]`
- [x] `pbmc_query` carries `cell_type` and `cell_subtype`, but the external note says final validation should be marker-based
- [x] `Vento` external query path recorded: `/home/data/fhz/project/phmap_package/data/real_test/ProjectSVR/query_data/decidua_query.h5ad`
- [x] `decidua_query` has count-like integer `adata.X` and no `layers["counts"]`
- [x] `decidua_query` currently exposes only `cell_type` on the verified query side

**Pending pairing / protocol fields**

- [ ] Quick-start converted query path: `TBD`
- [x] Reference sample-like grouping fields observed:
  - `mTCA`: `orig.ident`
  - `DISCO_hPBMCs`: `sample`
  - `cd4`: `sample`
  - `cd8`: `sample`
  - `vento`: `orig.ident`
- [ ] Domain key: `TBD`
- [ ] Final decision: which query-side labels, if any, are acceptable as evaluation truth vs marker-only review
- [ ] Final decision: whether `mTCA` needs a separately documented query before entering formal benchmark suite

**Planned scenarios**

- [x] External-validation pair: `DISCO_hPBMCs ref -> pbmc_query`
- [x] External-validation pair: `Vento ref -> decidua_query`
- [ ] Quick-start smoke benchmark (fast, tutorial-aligned)
- [ ] Full reference_atlas/query_data expansion (separate sub-plan after formal pairing/policy decisions are locked)

### 2.4 Dataset-specific scaling ceilings (record before manifest locking)

Principle: do **not** force every reference dataset through the full
`1k→50k` grid. Each dataset should only run the subset of training sizes that
remains feasible **after** the group-aware train/val/test split is materialized.

| Reference dataset | Total cells | Candidate training ceiling | Final admissible grid after split | Notes |
| --- | ---: | ---: | --- | --- |
| `PHMap_Lung_Full_v43_light` | 234,390 | `50k` | `TBD after split` | expected to support the full grid |
| `HLCA_Core` | 584,944 | `50k` | `TBD after split` | expected to support the full grid |
| `mTCA` | 188,862 | `50k` | `TBD after split` | expected to support the full grid |
| `DISCO_hPBMCs` | 167,594 | `50k` | `TBD after split` | expected to support the full grid |
| `cd4` | 150,361 | `50k` | `TBD after split` | expected to support the full grid |
| `cd8` | 225,212 | `50k` | `TBD after split` | expected to support the full grid |
| `Vento` | 70,325 | `<=50k` | `TBD after split` | likely needs a reduced upper ceiling after split |

Acceptance:

- [ ] Every reference dataset has a recorded admissible training-size list
- [ ] Any skipped size is documented as “not feasible after split”, not silently omitted

---

## 3) Phase 0 — Protocolize thresholds + dataset/shift suite (expert “must have”)

### P0.0 Locked benchmark design boundary

- [x] **Reference-heldout benchmark** is the only source of formal quantitative accuracy claims
- [x] **External query validation** is treated as unlabeled deployment-style evaluation unless query truth labels are explicitly approved later
- [x] Full-reference training is reserved for the final deployment model and is not used for benchmark reporting

### P0.0a Reference-heldout split protocol

Goal: compare methods on a fair, label-valid benchmark even when cross-dataset
annotation systems are not harmonized.

- [ ] For each reference dataset that has accepted truth labels, define a fixed split:
  - [ ] training pool
  - [ ] validation pool
  - [ ] held-out test pool
- [ ] Prefer **group-aware splitting** using the observed sample-like field:
  - [ ] PH-Map: `sample`
  - [ ] HLCA: `donor_id`
  - [ ] mTCA: `orig.ident`
  - [ ] DISCO_hPBMCs: `sample`
  - [ ] cd4: `sample`
  - [ ] cd8: `sample`
  - [ ] vento: `orig.ident`
- [ ] Preserve minimum label support after splitting (do not admit labels with too few cells into formal metrics)

Acceptance:

- [ ] No same-group leakage across train/val/test
- [ ] All reported benchmark labels remain supported in the held-out test set

### P0.0b Training-scale grid (reference size sensitivity)

Goal: test whether methods remain stable and resource-efficient as reference
size grows.

- [x] Locked training-size grid:
  - `1k`, `3k`, `6k`, `10k`, `20k`, `30k`, `40k`, `50k`
- [ ] For each dataset, use all training sizes up to the dataset-specific feasible maximum
- [ ] If the training pool is smaller than a target size, skip that size explicitly in the manifest/report

Acceptance:

- [ ] Every benchmark report clearly states which training sizes were admissible per dataset

### P0.0c Held-out evaluation-size grid (query/test scaling)

Goal: model practical query-time usage where modest query sizes are often
sufficient.

- [x] Locked held-out evaluation-size grid:
  - `1k`, `3k`, `5k`, `7k`, `10k`, `20k`
- [ ] Evaluate all methods on the same held-out test subsets
- [ ] Keep the full held-out test set as the anchor result when feasible

Acceptance:

- [ ] Scaling curves are derived from the same fixed held-out truth pool, not re-split ad hoc each time

### P0.0d External query validation policy

- [x] Query datasets are treated as unlabeled deployment-style inputs by default
- [x] Query-side author labels are not used as formal benchmark truth unless explicitly approved
- [x] Sankey plots are accepted as the primary label-correspondence visualization for query datasets with internal labels
- [ ] Marker-based validation panel is standardized:
  - [ ] marker dotplot / heatmap
  - [ ] optional module-score summaries
  - [ ] confidence / Unknown visualization on embeddings when available

Acceptance:

- [ ] No external query result is described as “accuracy” unless the truth-label approval has been explicitly recorded

### P0.0e Execution rollout policy (do not force a monolithic first pass)

Goal: make the benchmark executable in stages, because reference datasets do
not share identical label structures, sample-group columns, or feasible size
ceilings.

- [ ] Use **scenario-specific manifests**, not one universal benchmark manifest
- [ ] Use **reference-specific experiment dossiers**, not one monolithic cross-reference execution bundle
- [ ] Separate work into three rollout layers:
  - [ ] **Layer 1 — contract locking**
    - [ ] freeze split field, target labels, counts semantics, and domain key per dataset
    - [ ] create one manifest template for `reference_heldout`
    - [ ] create one manifest template for `external_query_validation`
  - [ ] **Layer 2 — pilot executable scenarios**
    - [ ] choose 2 reference-heldout pilots first (recommended: `PH-Map`, `HLCA`)
    - [ ] choose 1 ProjectSVR count-in-`adata.X` pilot (recommended: `DISCO_hPBMCs`)
    - [ ] choose 1 external-query visualization pilot (recommended: `PH-Map -> query_PH` or `HLCA -> hlca_query_GSE302339`)
  - [ ] **Layer 3 — scale expansion**
    - [ ] expand remaining reference datasets
    - [ ] add shift scenario(s)
    - [ ] add full training-size and held-out-size grids where feasible

Acceptance:

- [ ] The first execution pass is dataset-specific and produces at least one successful run before full-suite expansion
- [ ] No dataset is blocked by assumptions copied from another dataset's manifest
- [ ] Each reference dataset has its own code entrypoint/script set, output root, and summary notes

### P0.0f Fair comparison with limited per-reference tuning

Goal: allow each reference dataset to have its own executable experiment track
without collapsing fairness across methods.

- [ ] Lock a **shared comparator contract** across all references:
  - [ ] same primary target label for all methods inside one scenario
  - [ ] same split field and same held-out truth pool inside one scenario
  - [ ] same training-size / evaluation-size subset inside one scenario
  - [ ] same counts semantics and preprocessing declaration inside one scenario
- [ ] Allow **reference-specific tuning** only inside a bounded policy:
  - [ ] dataset-specific batch/domain key
  - [ ] dataset-specific feasible training ceiling
  - [ ] dataset-specific target label level
  - [ ] dataset-specific preprocessing metadata (gene namespace, counts source)
  - [ ] limited method hyperparameter adjustments when required by dataset scale or label structure
- [ ] Record every such adjustment in the comparator protocol table and scenario note

Acceptance:

- [ ] A reviewer can distinguish “fair shared protocol” from “dataset-specific operational tuning”
- [ ] No method receives an undocumented advantage on one reference dataset

### P0.1 Dataset pairs and shift definition (minimum evidence chain)

- [ ] Wire **≥2–3 dataset pairs** into benchmark manifests
- [ ] Add **≥1 explicit shift** scenario:
  - [ ] cross-dataset (PH-Map ↔ HLCA), or
  - [ ] platform/assay shift inside HLCA (domain-defined), or
  - [ ] label-set shift (hold out a set of fine types)
- [ ] For each formal scenario, define the manifest fields:
  - [ ] `split_name`, `split_description`
  - [ ] `reference_subset`, `query_subset`
  - [ ] `random_seed`
  - [ ] whether the scenario is `heldout_benchmark` or `external_query_validation`

Acceptance:

- [ ] Each scenario runs end-to-end and emits all required artifacts (Section 1).

### P0.2 Threshold selection rule (avoid cherry-picking)

Goal: Unknown/abstention thresholds are chosen by a **fixed rule**.

- [ ] Choose the primary rule and write it into protocol:
  - [ ] Rule A: pick threshold on validation to hit target coverage (e.g. 0.90)
  - [ ] Rule B: pick threshold that minimizes AURC on validation
- [ ] Emit the chosen threshold + rule into run metadata (`metrics.json`)
- [ ] Use the same thresholding rule for all AtlasMTL variants in a comparison

Acceptance:

- [ ] Re-running the same manifest reproduces the same threshold (within floating noise).

---

## 4) Phase A — Baseline gates + paper-grade curves

### A1 Baseline (CPU-light) gate (seeds + outputs)

- [ ] Run baseline AtlasMTL with:
  - [ ] `knn_correction="off"` (locked for early paper-grade evaluation)
  - [ ] `input_transform="binary"` (default)
  - [ ] HVG policy fixed (recorded)
- [ ] Run `seeds >= 3` (target 5)

Acceptance:

- [ ] All artifacts in Section 1 exist
- [ ] Report includes mean±std on lv4 endpoints

### A1b Reference-heldout benchmark table design

- [ ] For each accepted reference dataset, report:
  - [ ] per-level accuracy / macro-F1 / balanced accuracy
  - [ ] full-path metrics when hierarchical labels are used
  - [ ] calibration + selective prediction outputs
- [ ] For each training-size point, keep the same evaluation-size grid and group-aware split protocol

Acceptance:

- [ ] The main benchmark table is entirely reference-heldout, not mixed with external-query results

### A2 Calibration closure

- [ ] Run calibration toggle:
  - [ ] calibration off vs on (temperature scaling when val exists)
- [ ] Export:
  - [ ] ECE/Brier before vs after
  - [ ] reliability diagrams before vs after

Acceptance:

- [ ] ECE decreases without coverage collapse

### A3 Selective prediction closure

- [ ] Export risk–coverage curve + AURC (lv4)
- [ ] Export fixed coverage points (0.95/0.90/0.80) table rows

Acceptance:

- [ ] Curves exist for every formal run, not ad-hoc.

---

## 5) Phase B — Decision robustness (SWA/EMA)

This phase is evaluated only after Phase 0/Phase A baselines exist across the
dataset suite.

- [ ] B1 SWA on/off:
  - [ ] run `seeds=5` on at least 2 dataset pairs including a shift scenario
  - [ ] report mean±std for lv4 macro-F1 / balanced accuracy
- [ ] B2 EMA fallback (only if SWA fails)

Acceptance KPI:

- [ ] std decreases ≥25% on lv4 macro-F1/bAcc, without mean regression
- [ ] AURC and ECE do not worsen
- [ ] inference cost does not increase

---

## 6) Phase C — Long-tail and imbalance

- [ ] C1 class weights (none vs inv_freq vs effective_num)
- [ ] C2 focal loss (gamma/alpha grid)
- [ ] C3 balanced sampler
- [ ] C4 pick one default mechanism (avoid “stack everything”)

Acceptance KPI:

- [ ] tail-bin F1/recall improves on ≥2 dataset pairs
- [ ] AURC/ECE not worse than baseline

---

## 7) Phase D — Hierarchy-aware training and diagnostics

- [ ] D1 parent-conditioned decoding (flat vs conditioned)
- [ ] D2 hierarchy loss (lambda grid)
- [ ] D3 shift robustness check (run best hierarchy variant on shift scenario)

Acceptance KPI:

- [ ] cross-parent error rate decreases
- [ ] full-path covered accuracy improves
- [ ] coverage does not collapse

---

## 8) Phase E — CPU-first resource + scaling (formal requirement)

- [ ] Define scaling points per dataset (cells and HVG):
  - [x] training cells: `1k / 3k / 6k / 10k / 20k / 30k / 40k / 50k` (dataset-dependent ceiling)
  - [x] held-out evaluation cells: `1k / 3k / 5k / 7k / 10k / 20k`
  - [ ] HVG: at least 2 points (e.g. 3k / 6k)
- [ ] For each point, export:
  - [ ] train time, predict time
  - [ ] peak RSS
  - [ ] throughput (cells/s)
- [ ] Generate scaling plots

Acceptance:

- [ ] Scaling curves exist for at least 2 datasets (one shift-involving).

---

## 9) External query validation outputs (visualization-first)

- [ ] For each external query validation pair, export:
  - [ ] Sankey plot: predicted label vs query-side author label (when author labels exist)
  - [ ] marker-based dotplot / heatmap for predicted labels
  - [ ] confidence / Unknown distribution plot
  - [ ] embedding overlay when query has usable `obsm`
- [ ] Keep these outputs separate from the formal quantitative benchmark tables

Acceptance:

- [ ] External query results are presented as biological/operational validation, not as direct cross-dataset accuracy claims

---

## 10) Comparator fairness and reporting (continuous)

- [ ] For each comparator method run:
  - [ ] verify input semantics are recorded in `input_contract`
  - [ ] if a comparator cannot provide abstention curves, mark as “no selective outputs”
  - [ ] ensure Azimuth fallback is clearly labeled in metadata
- [ ] Fill resource reporting gaps:
  - [ ] if wrapper cannot report peak RSS/VRAM, tables must show `null` and the report must state “not reported by wrapper”

Acceptance:

- [ ] A reviewer can answer “was the comparison fair?” from protocol tables alone.

---

## 11) What to do when cleaned datasets arrive (handoff contract)

For each dataset, record the delivered cleaning notes in a short append-only
section (no code changes needed to start, only manifests + protocol metadata):

- [ ] cleaned `.h5ad` path(s)
- [ ] counts layer location and verification method
- [ ] `var_names_type` + species + mapping resource used
- [ ] label columns + any renaming/mapping applied
- [ ] domain_key choice and rationale
- [ ] known caveats (missing labels, partial levels, batch artifacts)

---

## 12) Immediate execution backlog (next practical step after plan update)

This section is the short-horizon rollout list to move from planning to actual
benchmark assets without overcommitting to one giant pipeline.

### 12.1 Immediate

- [ ] Freeze the scenario registry:
  - [ ] mark each scenario as `reference_heldout` or `external_query_validation`
  - [ ] record target label columns per scenario
  - [ ] record split field and candidate domain key per scenario
- [ ] Create one dossier root per reference dataset:
  - [ ] `PH-Map`
  - [ ] `HLCA`
  - [ ] `mTCA`
  - [ ] `DISCO_hPBMCs`
  - [ ] `cd4`
  - [ ] `cd8`
  - [ ] `Vento`
- [ ] Materialize manifest templates in repo:
  - [ ] `reference_heldout` template
  - [ ] `external_query_validation` template
- [ ] Lock output naming conventions:
  - [ ] manifest file names
  - [ ] output directory structure
  - [ ] run ID composition

### 12.2 Next

- [ ] Build the first pilot manifests:
  - [ ] `PH-Map` heldout single-level
  - [ ] `HLCA` heldout single-level
  - [ ] `DISCO_hPBMCs` heldout single-level
  - [ ] one external query validation manifest
- [ ] Build one script/output/report skeleton per pilot reference dossier
- [ ] Record dataset-specific admissible size grids after actual split materialization
- [ ] Record skipped sizes and the reason for each skipped size

### 12.3 Later

- [ ] Add multi-level reference-heldout manifests
- [ ] Add explicit shift scenarios
- [ ] Add deployment-only full-reference training manifests
