# Benchmark Ablation Discussion (2026-03-01)

This note captures the current benchmark design discussion and constraints so
future runs are consistent, paper-ready, and traceable.

## 1) Why AtlasMTL can be CPU-efficient

Although AtlasMTL is a deep-learning multi-task learning (MTL) model, it does
not automatically imply GPU is required or optimal.

Key reasons CPU can be competitive in our current regime:

- The current sampled runs are *short* (few epochs) and *moderate* scale.
  For such workloads, GPU overheads (initialization, kernel launch, host-device
  transfers, small-batch inefficiency) can dominate.
- The model family is an MLP encoder + classification heads (plus optional
  hierarchy enforcement and calibration). This is not inherently GPU-bound at
  the current sizes, and CPU multithreading often achieves good throughput.
- For many users, CPU-friendly runtime is a practical advantage: fewer hardware
  requirements, lower setup burden, and easier reproducibility.

Paper implication:

- CPU and GPU should be treated as separate benchmark variants for AtlasMTL.
  GPU is not assumed to be the default; it is an option whose benefit must be
  demonstrated in comparable runs.

## 2) Ablation priorities (what to test next)

This project is positioned primarily as reliable `sc -> sc` reference mapping
and multi-level label transfer. Ablations should focus on label accuracy,
reliability, hierarchy consistency, and resource efficiency.

### A) Task-weighting across label levels

We should test whether emphasizing fine-grained levels improves `anno_lv4`
accuracy and whether the effect is dataset-dependent.

- Example PH-Map weights:
  `task_weights = [0.3, 0.8, 1.5, 2.0]` for `[lv1, lv2, lv3, lv4]`
- Baseline uniform weights:
  `task_weights = [1, 1, 1, 1]`

Paper framing:

- Report both per-level metrics and (optionally) full-path metrics.
- Interpret weight sensitivity across datasets as evidence of either:
  - a generally preferable optimization bias toward fine labels, or
  - dataset-dependent tradeoffs that require protocol justification.

### B) Feature space: whole vs HVG

Current locked grid for the next round:

- HVG sizes:
  `3000/6000`
- Whole matrix:
  `whole`

Selection principle:

- The objective is not the single highest accuracy point.
- The formal goal is the best accuracy-resource balance under the current
  research positioning.
- In practice, this means:
  - keep `anno_lv4` accuracy and macro-F1 as primary quality endpoints
  - keep training time, prediction time, peak RSS, and peak GPU memory as
    required resource endpoints
  - prefer the lowest-resource candidate among runs that remain close to the
    best quality result
- Therefore, `whole` must remain the strong baseline, while HVG settings are
  tuned as operational tradeoff candidates rather than being treated as a
  guaranteed universal improvement

Constraints:

- HVG selection must be reference-derived after canonicalization.
- HVG selection must use `layer="counts"` under `hvg_method="seurat_v3"`.
- HVG > ~6000 may be diminishing returns; this should be demonstrated (or
  rejected) empirically via accuracy–resource curves.

Recommended follow-up HVG search protocol:

- hold `input_transform`, `task_weights`, seeds, and train config fixed
- compare `whole`, `hvg3000`, `hvg4000`, `hvg5000`, `hvg6000`, `hvg7000`,
  `hvg8000`
- identify the best observed quality run
- define a near-optimal band around that quality target
- select the lowest-resource run inside that band as the dataset-level
  recommended HVG setting

### C) Input transform: binary vs non-binary

Binary input is a key design choice with a plausible tradeoff:

- potential generalization/robustness benefits
- potential fine-grained signal loss (hurting `anno_lv4`)
- significant compute/memory differences

Requirement:

- Any binary-vs-non-binary comparison must report *both* prediction quality and
  resource usage. It is insufficient to report accuracy only.
- For the next formal ablation round, `non-binary` is defined narrowly as
  `raw counts -> float32`, not `log1p(counts)` and not log-normalized `X`.
- AtlasMTL ablations should source both `binary` and `float` inputs from the
  same `layers["counts"]` matrix so the transform itself remains the only
  changing variable.

### D) CUDA gate before GPU benchmarking

GPU benchmarking must pass an execution-environment gate before it is admitted
into the formal ablation matrix.

- User-interactive CUDA availability is not sufficient by itself.
- The same non-interactive benchmark entrypoint must pass:
  - `torch.cuda.is_available()`
  - `torch.cuda.device_count() > 0`
  - a minimal AtlasMTL train/predict smoke on `device="cuda"`
- If the gate fails, CPU remains the formal benchmark result and GPU is
  documented as environment-unverified for that run.

## 3) What to report: metrics and resources

### Label-transfer performance (primary)

For shared target level (e.g. `anno_lv4`) and for multi-level runs:

- accuracy
- macro-F1
- balanced accuracy
- coverage / reject rate (if abstention is enabled)
- calibration metrics (ECE, Brier) when probability outputs are available
- risk/AURC when coverage-abstention behaviors are studied

AtlasMTL-specific secondary:

- hierarchy path consistency metrics
- full-path accuracy / coverage (when hierarchy rules are supplied)
- KNN rescue/harm metrics (only when coordinate inputs exist)

### Resource usage (paper-facing)

For each method and for each AtlasMTL variant (CPU/GPU, transform, HVG size):

- train elapsed seconds
- predict elapsed seconds
- throughput (items/s)
- peak RSS (GB)
- average RSS (GB)
- peak GPU memory (GB) when applicable
- average GPU memory (GB) when applicable
- device used (`cpu` / `cuda`)
- configured thread count (and, when available, CPU core-equivalent usage)

Reporting rule:

- CPU and GPU AtlasMTL results should be separate rows/variants in tables and
  figures. Do not merge them into a single “atlasmtl” row.

## 4) Fairness / comparability rules

- Shared-target comparison comes first (single-level `anno_lv4`), even if
  AtlasMTL supports multi-level capabilities.
- External comparators are treated as single-level baselines unless they
  natively support richer contracts.
- Any comparison must state the input contract (matrix source, counts layer,
  normalization mode, feature alignment). Protocol tables are mandatory.
- Resource comparisons must be aligned on the same reference/query dataset and
  must not mix CPU and GPU variants in the same row.

## 5) Execution guidance (practical)

- Maintain an explicit “ablation matrix” manifest that enumerates:
  - `task_weights`
  - `feature_space` + `n_top_genes` (if HVG)
  - `input_transform`
  - `device`
- Generate paper tables from a single `metrics.json` bundle per ablation run,
  and keep runtime artifacts private under `~/tmp/`.

## 6) Current conclusion from the completed ablation round

The completed AtlasMTL ablation supports a pragmatic default-selection rule:

- `hvg6000 + binary + phmap` is the current best default candidate because it
  improves the accuracy-resource balance
- this does not mean `hvg6000` is now a fixed universal default for all future
  datasets
- future benchmark rounds should keep `whole` as the anchor baseline and use a
  local HVG grid to identify the dataset-level operational optimum

## 7) Current conclusion from the KNN correction round

The completed formal CPU KNN run using external `X_scANVI` space supports a
different conclusion:

- KNN correction should not be enabled by default
- `low_conf_only` is the only KNN mode worth retaining as a secondary ablation
- current evidence does not support KNN as a reliable source of better overall
  final accuracy

Interpretation:

- KNN can improve some class-balance-oriented metrics
- KNN can improve covered / accepted-sample accuracy
- but KNN also reduces coverage and introduces a measurable harm rate
- in the current run, these tradeoffs do not justify making KNN part of the
  default AtlasMTL protocol

Operational recommendation:

- keep KNN implemented and benchmarked
- keep `knn_correction="off"` as the primary recommended setting
- reserve `knn_correction="low_conf_only"` for controlled method-specific
  follow-up analysis
