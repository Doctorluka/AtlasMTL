# AtlasMTL Technical Report (for expert consultation)

Date: **2026-03-02**  
Audience: **computer science / ML systems expert**  
Scope: framework design + completed experiments + optimization outcomes + next optimization directions.

> This is an engineering report, not a paper-claim document. Results are from
> sampled real-data bundles and internal ablations.

## 0. Executive summary

**What AtlasMTL is:** a single-cell **sc→sc reference mapping** framework focused on **reliable multi-level label transfer** (accuracy + calibration + abstention + hierarchy consistency + traceability), rather than primarily competing on generic integrated embeddings.

**Current state (sampled real benchmark):**

- Performance headroom concentrates at fine labels (`anno_lv4`), while coarse labels are near-saturated.
- KNN correction is implemented and benchmarked, but **does not show stable net accuracy gains** and can be harmful depending on geometry; default inference remains **`knn_correction="off"`**.
- Ablations show stable wins for: `input_transform="binary"`, **phmap-like task weights**, and **HVG** feature space (better resource tradeoff).

**Most recommended next optimization direction:** improve the **MTL classifier itself** (imbalance + hierarchy-aware training + representation regularization + confidence quality), and improve **decision robustness** with **single-model SWA** (low deployment overhead).

Key sources:

- Positioning & architecture: `documents/design/overall_summary.md`, `documents/design/architecture.md`, `documents/design/research_positioning.md`
- Stable API contract: `documents/design/api_contract.md`
- Sampled real benchmark summaries: `documents/experiments/2026-03-01_real_mapping_benchmark/results_summary/*`
- Ablations & recommendations: `documents/experiments/2026-03-01_real_mapping_benchmark/ablation_optimization/results_summary/*`
- KNN failure analysis (with numbers):  
  `documents/experiments/2026-03-01_real_mapping_benchmark/ablation_optimization/knn_eval/*_internal_discussion.md`
- MTL optimization plan (next cycle):  
  `documents/experiments/2026-03-02_mtl_classifier_optimization/plan/2026-03-02_execution_plan.md`  
  `documents/experiments/2026-03-02_mtl_classifier_optimization/notes/2026-03-02_decision_robustness_discussion.md`

## 1. Problem framing & research positioning

AtlasMTL answers:

> Given a labeled reference atlas and an unlabeled query dataset, can we assign
> accurate and reliable labels (multi-level), **calibrated** confidence signals,
> optional **Unknown/abstention**, and **hierarchically consistent** predictions,
> while keeping end-to-end runs reproducible and auditable?

This positioning intentionally prioritizes label-transfer quality and reliability
over scIB-style “best integrated embedding” endpoints. See:
`documents/design/research_positioning.md`.

## 2. Framework overview (AnnData contract & user-facing interface)

### 2.1 AnnData in, AnnData out

AtlasMTL is built around the scverse contract:

- input: `AnnData` reference + query
- output: predictions written back into query `AnnData` (plus export tables)

### 2.2 Stable high-level Python API

The stable API surface is intentionally thin (to keep user contract stable while
internals evolve):

- `build_model(...) -> TrainedModel`
- `predict(model, adata_query, ...) -> PredictionResult`

Full contract and writeback fields are documented in:
`documents/design/api_contract.md`.

### 2.3 What gets written back (audit-friendly)

Per label level `<level>`:

- `obs["pred_<level>"]`
- `obs["conf_<level>"]`, `obs["margin_<level>"]`
- `obs["is_unknown_<level>"]` (abstention)

Optional coordinates:

- `obsm["X_pred_latent"]`, `obsm["X_pred_umap"]` (when configured)

Metadata:

- `uns["atlasmtl"]` stores settings, runtime summaries, calibration info, etc.

## 3. Architecture & design rationale (why it is structured this way)

### 3.1 “Thin API, modular internals”

AtlasMTL splits responsibilities by volatility and audit needs:

- `atlasmtl/core/`: training + prediction + runtime summaries + stable types  
  **Rationale:** core behavior must be testable and stable; it defines user-visible semantics.
- `atlasmtl/mapping/`: calibration, KNN logic, open-set/Unknown, hierarchy enforcement  
  **Rationale:** these policies iterate frequently; isolating them reduces churn in core training code.
- `atlasmtl/models/`: artifact serialization, manifest, checksums, reference storage  
  **Rationale:** reproducibility depends on artifacts; making them first-class prevents “model file drift”.
- `atlasmtl/io/`: AnnData writeback/export helpers  
  **Rationale:** keeps I/O contract stable and prevents training code from mixing export semantics.
- `benchmark/`: protocolized comparator runners + harmonized outputs  
  **Rationale:** benchmark is a separate product: fairness, traceability, and resource accounting matter.

See design overview: `documents/design/overall_summary.md`, `documents/design/architecture.md`.

### 3.2 Why calibration + abstention are first-class

Reference mapping faces distribution shift and novel cell states. A pure closed-set classifier
will confidently mislabel unknowns. Therefore:

- post-hoc calibration (temperature scaling) is implemented and supported
- “two-layer evaluation” is used: end-to-end accuracy + covered accuracy/coverage under Unknown

### 3.3 Why hierarchy consistency is a primary constraint

Multi-level labels imply parent–child constraints. Independent flat classifiers can produce
logical contradictions. AtlasMTL therefore evaluates and enforces hierarchy consistency and
tracks full-path metrics (accuracy/coverage/covered accuracy).

### 3.4 Why gene-ID policy and counts contract are explicit

AtlasMTL relies on exact feature alignment, making gene-ID canonicalization and counts layer
contract critical. Formal runs should record:

- `var_names_type` (symbol vs ensembl)
- species
- mapping resource used
- duplicate/unmapped gene policy
- raw counts source (`layers["counts"]`)

See: `documents/design/overall_summary.md`, `documents/design/preprocessing.md`.

## 4. Artifact & traceability contract

Preferred artifact layout (external reference storage):

- `model.pth`
- `model_metadata.pkl`
- `model_reference.pkl`
- `model_manifest.json`

Artifacts are intended to be auditable and benchmark-friendly (checksums, sizes, configs).
See: `documents/design/model_artifacts.md`.

> Historical note: this report was written before the comparator refactor on `2026-03-03`.
> Mentions of `azimuth` and `seurat_anchor_transfer_fallback` should now be interpreted as the predecessor
> of the formal `seurat_anchor_transfer` comparator.

## 5. Benchmark framework: comparator closure and protocol choices

Comparator integration work (as of 2026-02-28) established a unified benchmark runner and
method wrappers for:

- `celltypist`, `scanvi`, `singler`, `symphony`, `azimuth` (+ fallback labeling at that time)
- plus local baselines: `atlasmtl`, `reference_knn`

Comparator scope decision: external methods are treated as single-level or per-level baselines
unless they natively support multi-level hierarchy. See:
`documents/tmp/2026-02-28_celltypist_benchmark_progress_report.md`.

## 6. Completed experiments and quantitative summaries

### 6.1 2026-03-01 sampled real-data benchmark: data audit & preprocessing

Datasets:

- reference: `data/test_adatas/sampled_adata_10k.h5ad` (10k × 21977)
- query: `data/test_adatas/sampled_adata_3000.h5ad` (3k × 21977)

Key facts:

- `layers["counts"]` present and treated as authoritative raw counts
- `obsm` empty → coordinate/KNN-by-coords not supported in this specific run
- gene IDs: symbol → versionless Ensembl via bundled BioMart table  
  (`atlasmtl/resources/gene_id/biomart_human_mouse_rat.tsv.gz`)

Canonicalization outcome (both reference & query):

- input genes: 21977
- canonical kept: 21510
- unmapped dropped: 467
- duplicates collapsed: 0

Source: `documents/experiments/2026-03-01_real_mapping_benchmark/results_summary/raw_data_audit.md`.

### 6.2 Single-level comparator benchmark (target = `anno_lv4`)

Headline metrics (accuracy / macro-F1 / balanced accuracy):

| method | accuracy | macro_f1 | balanced_accuracy | note |
| --- | ---: | ---: | ---: | --- |
| celltypist | 0.7900 | 0.7093 | 0.7104 | strongest on this sampled bundle |
| atlasmtl | 0.7467 | 0.6381 | 0.6068 | strongest learned-in-runner non-external baseline |
| azimuth | 0.7343 | 0.5888 | 0.5854 | historical predecessor of current `seurat_anchor_transfer` |
| singler | 0.6850 | 0.5990 | 0.6360 | competitive classical baseline |
| symphony | 0.6160 | 0.4838 | 0.4875 | completed |
| scanvi | 0.5773 | 0.3161 | 0.3409 | completed |
| reference_knn | 0.3630 | 0.2917 | 0.2606 | weak baseline |

Metrics source:
`~/tmp/atlasmtl_real_mapping_benchmark_20260301/single_level_benchmark/all_methods_final_v2/metrics.json`
and summarized in:
`documents/experiments/2026-03-01_real_mapping_benchmark/results_summary/single_level_benchmark_summary.md`.

Important caveat:

- this is a sampled real-data bundle; not paper-final
- KNN correction disabled here
- Azimuth result is explicitly fallback-mode and should not be presented as strict native Azimuth

### 6.3 Multi-level AtlasMTL (anno_lv1..anno_lv4) + hierarchy metrics

Per-level metrics:

| level | accuracy | macro_f1 | balanced_accuracy |
| --- | ---: | ---: | ---: |
| anno_lv1 | 0.9877 | 0.9855 | 0.9858 |
| anno_lv2 | 0.9353 | 0.8569 | 0.8822 |
| anno_lv3 | 0.8700 | 0.7473 | 0.7477 |
| anno_lv4 | 0.7437 | 0.6274 | 0.5990 |

Hierarchy metrics:

- edge consistency: all edges = 1.0000
- full-path accuracy: 0.7330
- full-path coverage: 0.8993
- covered full-path accuracy: 0.8150

Source:
`~/tmp/atlasmtl_real_mapping_benchmark_20260301/multilevel_atlasmtl/metrics.json`
and summarized in:
`documents/experiments/2026-03-01_real_mapping_benchmark/results_summary/multilevel_atlasmtl_summary.md`.

Interpretation:

- coarse levels are near-saturated; fine labels (`anno_lv4`) are the main headroom
- hierarchy enforcement is functioning and auditable

### 6.4 AtlasMTL ablation optimization (24 variants, KNN off)

Scope:

- device: cpu vs cuda
- feature space: whole vs hvg3000 vs hvg6000
- input transform: binary vs float
- task weights: uniform vs phmap

Top variants by `anno_lv4` accuracy (selected):

1. `cuda + hvg6000 + binary + phmap`: accuracy 0.7730, macro-F1 0.6720
2. `cpu + hvg6000 + binary + phmap`: accuracy 0.7677, macro-F1 0.6618

Stable observations:

- `binary` >> `float`
- `phmap` weights > uniform
- HVG improves resource tradeoff; `hvg6000` looked best in this family

Resource highlights (representative):

- peak RSS: whole ≈ 4.84–5.06 GB; hvg6000 ≈ 3.42 GB
- average train time: cpu ≈ 10.75s; cuda ≈ 3.43s (sampled bundle)

Source:
`documents/experiments/2026-03-01_real_mapping_benchmark/ablation_optimization/results_summary/atlasmtl_ablation_summary.md`.

### 6.5 HVG tradeoff & weight scan recommendations (internal)

HVG tradeoff recommendation:

- cpu: hvg5000
- cuda: hvg6000

Weight scan recommendation:

- cpu: `ratio_1.6`
- cuda: `lv4strong_a = [0.2, 0.7, 1.5, 3.0]`

Source:
`documents/experiments/2026-03-01_real_mapping_benchmark/ablation_optimization/results_summary/hvg_tradeoff_recommendation.md`  
`documents/experiments/2026-03-01_real_mapping_benchmark/ablation_optimization/results_summary/weight_scan_recommendation.md`

### 6.6 KNN correction evaluation: what failed and why (with numbers)

**Formal external-space run (`X_scANVI`) summary:** KNN does not support enabling as default;
it behaves as a selective tradeoff (macro/balanced metrics sometimes up, but accuracy/coverage/unknown/harm tradeoffs are non-trivial).

**Geometry evaluation (real deployment assumption: query has expression only):**

Main lv4 table (no-obsm query):

| geometry_mode | knn_variant | accuracy_lv4 | covered_acc_lv4 | coverage_lv4 | unknown_rate_lv4 | macro_f1_lv4 | balanced_acc_lv4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| predicted_scanvi_head | knn_off | 0.7683 | 0.8294 | 0.9263 | 0.0737 | 0.6913 | 0.6589 |
| predicted_scanvi_head | knn_lowconf | 0.6557 | 0.8673 | 0.7560 | 0.2440 | 0.6096 | 0.5281 |
| predicted_scanvi_head | knn_all | 0.0603 | 0.0810 | 0.7447 | 0.2553 | 0.0141 | 0.0154 |
| latent_internal | knn_off | 0.7657 | 0.8201 | 0.9337 | 0.0663 | 0.6728 | 0.6387 |
| latent_internal | knn_lowconf | 0.7527 | 0.8492 | 0.8863 | 0.1137 | 0.6797 | 0.6324 |
| latent_internal | knn_all | 0.7533 | 0.8458 | 0.8907 | 0.1093 | 0.6848 | 0.6414 |

Key conclusion:

- geometry matters: `latent_internal` is far more stable than `predicted_scanvi_head`
- but neither geometry shows net end-to-end accuracy gain over `knn_off` under current policy
- therefore default inference remains `knn_correction="off"`; keep only limited KNN ablations on `latent_internal`

Source:
`documents/experiments/2026-03-01_real_mapping_benchmark/ablation_optimization/knn_eval/internal_discussion.md`  
`documents/experiments/2026-03-01_real_mapping_benchmark/ablation_optimization/knn_eval/geometry_eval_internal_discussion.md`

## 7. What improved vs what failed (so far)

### 7.1 Confirmed stable improvements / wins

- input transform: `binary` consistently outperforms `float`
- task weighting: `phmap`-style weights improve `anno_lv4`
- feature space: HVG provides better resource/quality tradeoffs than whole-matrix
- calibration: temperature scaling exists and is compatible with the evaluation contract
- benchmark engineering: comparator closure and protocolized outputs exist; resource accounting is being standardized (but see limitations)

### 7.2 Failures / limitations / non-closure items

- KNN correction: no stable net accuracy gain; can be harmful depending on geometry and thresholds → keep off by default
- predicted coordinate head + KNN: catastrophic in tested setting (`knn_all`) and high harm even for `lowconf`
- sampled datasets without `obsm`: not sufficient for coordinate regression evaluation in the original benchmark run
- comparator resource reporting: many wrappers still have incomplete peak RSS fields (`null`), limiting paper-grade resource tables
- historical Azimuth predecessor completed via fallback (`seurat_anchor_transfer_fallback`); current formal comparator is `seurat_anchor_transfer`

## 8. Main optimization direction: improve the MTL classifier itself (next cycle)

Rationale:

- coarse levels are near-saturated; main headroom is at `anno_lv4`
- KNN is not a reliable default booster; the dominant lever is core classification robustness

This direction is formalized as the next experiment dossier:
`documents/experiments/2026-03-02_mtl_classifier_optimization/`.

### 8.1 Phase A (no-code tuning): low-risk levers

From `documents/experiments/2026-03-02_mtl_classifier_optimization/plan/2026-03-02_execution_plan.md`:

- train time + early stopping: grid over `num_epochs`, `val_fraction`, `patience`
- architecture: tune `hidden_sizes` and `dropout_rate`
- preprocessing: HVG size tradeoff with `whole` as stability anchor
- task weighting: validate stronger `lv4` emphasis, and **re-validate on a second dataset or split**

### 8.2 Phase B (code changes): prioritized by expected ROI

1) address class imbalance at fine levels (`anno_lv4`):
   - per-class weighting / focal loss
   - class-balanced sampling
2) make hierarchy consistency part of training (not only post-processing):
   - hierarchical constraint loss / parent-conditioned decoding
3) improve representations for fine labels:
   - supervised contrastive / metric-learning regularizers on latent
4) improve confidence quality:
   - calibration by default when validation exists
   - optional label smoothing

### 8.3 Decision robustness (reduce single-run stochasticity): single-model SWA (preferred)

Goal: stable decisions with low resource overhead (CPU-first deployment).

Locked preference and acceptance criteria are documented in:
`documents/experiments/2026-03-02_mtl_classifier_optimization/notes/2026-03-02_decision_robustness_discussion.md`.

Key points:

- prioritize **single-model SWA** (EMA as fallback) to reduce training stochasticity
- keep as experiment toggle first; do not change defaults until validated
- accept SWA if N=5 seeds show std decreases ≥ 25% on lv4 macro-F1/balanced-acc, with no mean regression and no coverage collapse, and no inference-cost increase

## 9. Completeness check (is this the full set? what may be missing)

### 9.1 Covered in this report (repo-tracked)

- framework design / positioning / API & artifacts (design docs)
- comparator closure design and tests (2026-02-28 progress report)
- sampled real benchmark dossier (single-level + multi-level)
- AtlasMTL-only ablations + HVG/weight recommendations
- KNN failure analysis including geometry-based evaluations
- next-cycle MTL optimization plan and decision-robustness strategy (SWA)

### 9.2 Likely remaining gaps / missing closures (as of 2026-03-02)

1) **Second dataset (or alternate split) validation** to test generalization of:
   - HVG recommendations
   - task weighting (lv4-strong)
   - any new loss/sampler/hierarchy-training change
2) **Comparator backend clarification** if needed for formal comparator claims (`seurat_anchor_transfer` backend should be reported explicitly)
3) **Comparator resource accounting completeness** (peak/avg RSS, CPU usage) to produce paper-grade resource tables
4) **Coordinate regression evaluation** on datasets that actually contain suitable `obsm` targets (for non-KNN claims)
5) **KNN evaluation across datasets** if KNN is to be discussed beyond an ablation (currently evidence is insufficient)
