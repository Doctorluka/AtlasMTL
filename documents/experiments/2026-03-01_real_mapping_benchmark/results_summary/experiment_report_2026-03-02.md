# AtlasMTL Experiment Report (2026-03-02)

This report consolidates the current AtlasMTL experiment round under the
project positioning "sc->sc reference mapping and multi-level label transfer".
It is an internal engineering report intended to guide the next optimization
cycle. It is not a finalized paper claim document.

## 0) Scope and artifacts

### Datasets used in this round

Sampled real-data benchmark bundle:

- reference: `data/test_adatas/sampled_adata_10k.h5ad`
- query: `data/test_adatas/sampled_adata_3000.h5ad`

KNN / coordinate-enabled bundle (with `obsm`):

- reference: `data/test_adatas/knn/reference_10k.h5ad`
- query: `data/test_adatas/knn/query_3k.h5ad`

### Where runtime outputs live

As per repository policy, large run outputs live in the user's private
workspace under `~/tmp/` and are referenced from repo docs:

- single-level all-method bundle:
  `~/tmp/atlasmtl_real_mapping_benchmark_20260301/single_level_benchmark/all_methods_final_v2/`
- multi-level AtlasMTL run:
  `~/tmp/atlasmtl_real_mapping_benchmark_20260301/multilevel_atlasmtl/`
- ablation optimization grid + plots:
  `~/tmp/atlasmtl_real_mapping_benchmark_20260301/ablation_optimization/`
- formal KNN scanvi-space run:
  `~/tmp/atlasmtl_knn_scanvi_eval_20260302_cpu_formal_v2/`
- KNN geometry evaluation (real-query simulation via stripping `obsm/obsp`):
  `~/tmp/atlasmtl_knn_geometry_eval_20260302_cpu_v2/`

## 1) What was completed

### 1.1 Single-level comparator benchmark

Completed methods: `atlasmtl`, `reference_knn`, `celltypist`, `scanvi`,
`singler`, `symphony`, and `azimuth` (with explicit fallback labeling).

Summary and interpretation live in:

- `documents/experiments/2026-03-01_real_mapping_benchmark/results_summary/single_level_benchmark_summary.md`

### 1.2 Multi-level AtlasMTL run

Completed AtlasMTL MTL run on `anno_lv1..anno_lv4` with hierarchy enforcement.
Per-level metrics and hierarchy checks are recorded in:

- `documents/experiments/2026-03-01_real_mapping_benchmark/results_summary/multilevel_atlasmtl_summary.md`

### 1.3 AtlasMTL ablation optimization

Completed a 24-variant grid (device, feature space, input transform, weights)
with KNN disabled, yielding stable conclusions about:

- `binary` vs `float`
- `phmap` vs `uniform` task weights
- `whole` vs `hvg` feature selection
- CPU vs CUDA runtime and memory

See:

- `documents/experiments/2026-03-01_real_mapping_benchmark/ablation_optimization/results_summary/atlasmtl_ablation_summary.md`
- `documents/experiments/2026-03-01_real_mapping_benchmark/ablation_optimization/results_summary/interim_hvg_weight_comparison.md`
- `documents/experiments/2026-03-01_real_mapping_benchmark/ablation_optimization/results_summary/hvg_tradeoff_recommendation.md`
- `documents/experiments/2026-03-01_real_mapping_benchmark/ablation_optimization/results_summary/weight_scan_recommendation.md`

### 1.4 KNN correction evaluation (two rounds)

Two complementary KNN evaluations were executed:

1. formal CPU KNN run using *external* `obsm["X_scANVI"]` space on both
   reference and query (not a real deployment assumption).
2. KNN geometry re-evaluation under the real deployment assumption: query has
   only an expression matrix, and any KNN query coordinates must come from the
   model output.

See internal notes:

- `documents/experiments/2026-03-01_real_mapping_benchmark/ablation_optimization/knn_eval/internal_discussion.md`
- `documents/experiments/2026-03-01_real_mapping_benchmark/ablation_optimization/knn_eval/geometry_eval_internal_discussion.md`

## 2) KNN correction failure analysis (what failed and why)

### 2.1 "KNN helped balance metrics but not accuracy"

In the formal CPU `X_scANVI` run, enabling KNN tended to:

- improve some class-balance metrics (macro-F1 / balanced accuracy)
- increase `Unknown` rate and reduce coverage
- introduce non-trivial harm rate (overwriting correct MTL predictions)

This pattern indicates KNN behaves more like a selective tradeoff mechanism
(reject/rescue) than a reliable net accuracy booster under the current policy.

### 2.2 Geometry is the critical decision point

The geometry eval separated two strategies:

- A) `predicted_scanvi_head`: learn a coordinate regression head against
  reference `X_scANVI`, predict query coordinates from expression, then KNN in
  the predicted scanvi space.
- B) `latent_internal`: no coordinate head; use encoder latent space for KNN.

Empirical outcome:

- A is currently *not viable* for KNN correction:
  even `low_conf_only` produced a large harm rate; `knn_all` was catastrophic.
- B is comparatively stable:
  KNN does not improve end-to-end accuracy over `knn_off`, but it improves
  covered accuracy at the cost of reduced coverage.

Interpretation:

- Coordinate regression quality metrics can look acceptable while KNN voting
  remains harmful. KNN depends on *label neighborhood purity* more than global
  geometric plausibility.
- The predicted scanvi space is not guaranteed to preserve "same-label local
  neighborhoods" for the mapping setting and label set; therefore KNN voting
  is an unreliable post-processing layer in that geometry.

### 2.3 Why KNN does not yield net accuracy gains (current hypothesis)

Even when geometry is reasonably aligned (e.g. `latent_internal`), KNN can fail
to improve end-to-end accuracy because:

- It targets hard/borderline cells, where local neighborhoods are inherently
  less label-pure.
- It can overwrite correct MTL predictions (harm) unless the gating thresholds
  are extremely conservative.
- When coupled with an abstention/Unknown policy, coverage drops can offset
  improvements in covered accuracy.

### 2.4 Current decision (project positioning aligned)

Given the evidence in this round:

- Default inference should remain `knn_correction="off"`.
- KNN remains a method-specific ablation only, and should be restricted to the
  `latent_internal` branch for future exploration.

This is aligned with the positioning: label mapping accuracy/reliability first,
and avoids anchoring the method narrative on a module without stable gains.

## 3) Auxiliary optimization paths (what else we can improve without changing the core claim)

These are supported by completed ablations and are likely to give stable value:

### 3.1 Input and feature choices

- Prefer `input_transform="binary"` over `"float"` for this dataset family.
- Keep `whole` as a stability baseline in search grids, but operationally
  prefer HVG for memory/runtime.
- Current search support suggests `hvg5000` on CPU and `hvg6000` on CUDA are
  good operational candidates, but should not be frozen as universal defaults.

### 3.2 Task-weighting

- `phmap` weights consistently outperform uniform weighting on `anno_lv4`.
- Weight scan suggests stronger fine-level emphasis can be valid, but must be
  validated on additional datasets before being made a project default.

### 3.3 Calibration and abstention policy

- Temperature scaling calibration is implemented and can improve reliability
  metrics without changing the core label-transfer objective.
- "Two-layer evaluation" (end-to-end accuracy plus covered accuracy/coverage)
  is the correct reporting approach when Unknown/reject behavior exists.

### 3.4 Resource accounting standardization

- AtlasMTL resource reporting is already strong (elapsed + peak RSS).
- Comparator wrappers should be extended to standardize peak/average memory
  reporting to support paper-grade resource comparisons.

## 4) Main model optimization: where the real headroom likely is

The current ablation results suggest we should prioritize improving the MTL
classifier itself, because that is the dominant contributor to performance and
robustness. The biggest headroom is expected at `anno_lv4` (fine labels).

### 4.1 Confirmed optimization levers already supported by the code

These can be tuned without architectural changes:

- train time: increase `num_epochs` with a validation split (`val_fraction`)
  and early stopping (`early_stopping_patience`)
- architecture: tune `hidden_sizes` and `dropout_rate`
- preprocessing: keep counts contract stable, and evaluate HVG sizes under the
  resource+quality tradeoff objective
- task weighting: validate strong-lv4 weights on additional datasets

### 4.2 Likely next model improvements (requires code changes)

Proposed priorities, ordered by expected ROI and interpretability:

1. Address class imbalance at fine levels:
   - per-class weighting or focal loss at `anno_lv4`
   - class-balanced sampling in training batches
2. Make hierarchy consistency part of training (not only post-processing):
   - explicit hierarchical constraint loss or parent-conditioned decoding
   - ensure this does not inflate Unknown by construction
3. Improve representation quality for fine labels:
   - metric-learning style regularizers (contrastive / supervised contrastive)
     on the latent space to improve label neighborhood purity without KNN
4. Improve confidence quality:
   - calibration by default when validation is enabled
   - optional label smoothing for better probability estimates

### 4.3 What not to prioritize right now

- KNN correction expansion: current evidence is not strong and the module adds
  interpretability/maintenance burden.
- Embedding integration metrics as primary endpoints: outside current
  positioning.

## 5) Immediate next actions

1. Lock default inference as `knn_correction="off"` and keep KNN as an ablation
   on `latent_internal` only.
2. Choose an operational "best known" AtlasMTL config for the next dataset:
   - `binary + HVG + phmap-like weights`, then re-validate.
3. Run a second dataset (or split) to test whether:
   - HVG and weight recommendations generalize
   - `anno_lv4` improvements persist across different label distributions
4. Start the next model-optimization cycle focusing on imbalance + hierarchy
   training constraints + calibration.

