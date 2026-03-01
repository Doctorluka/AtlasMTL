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

We should benchmark a grid such as:

- HVG sizes:
  `1000/2000/3000/4000/5000/6000`
- Whole matrix:
  `whole`

Constraints:

- HVG selection must be reference-derived after canonicalization.
- HVG selection must use `layer="counts"` under `hvg_method="seurat_v3"`.
- HVG > ~6000 may be diminishing returns; this should be demonstrated (or
  rejected) empirically via accuracy–resource curves.

### C) Input transform: binary vs non-binary

Binary input is a key design choice with a plausible tradeoff:

- potential generalization/robustness benefits
- potential fine-grained signal loss (hurting `anno_lv4`)
- significant compute/memory differences

Requirement:

- Any binary-vs-non-binary comparison must report *both* prediction quality and
  resource usage. It is insufficient to report accuracy only.

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
