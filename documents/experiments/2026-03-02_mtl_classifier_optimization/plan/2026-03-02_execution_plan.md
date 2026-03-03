# Execution Plan (MTL classifier optimization)

This plan records the concrete next experiments derived from the MTL-focused
optimization section of:

- `documents/experiments/2026-03-01_real_mapping_benchmark/results_summary/experiment_report_2026-03-02.md`

## 0) Baseline assumptions to keep fixed

Locked decisions for this cycle (unless explicitly stated in a sub-experiment):

- inference: `knn_correction="off"`
- input transform: `binary`
- features: HVG-based by default, keep `whole` as a stability control point
- task weighting: start from `phmap-like` weights, allow targeted `lv4` upweight
- evaluation: always report both end-to-end and covered metrics when Unknown is enabled

## 1) Primary endpoints (must report)

Primary focus is `anno_lv4` (fine labels) while keeping hierarchy and upper
levels stable.

Minimum metric set:

- per-level: `accuracy`, `macro_f1`, `balanced_accuracy` for `anno_lv1..anno_lv4`
- hierarchy: edge consistency, full-path `accuracy`, full-path `coverage`,
  covered full-path accuracy
- abstention: `Unknown` rate (and any reject/gating stats if enabled)
- calibration (when validation exists): ECE or Brier score (per level if available)

Hard constraints (stop/reject a candidate if violated):

- no meaningful regression on `anno_lv1..anno_lv3` beyond noise
- no "fake progress" where full-path coverage collapses to gain covered accuracy

## 2) Common training protocol

- validation: stratified `val_fraction` (prefer stratification by `anno_lv4`)
- early stopping: enabled; record best epoch
- seeds: run >= 3 seeds for any candidate that looks promising
- artifact policy: store large outputs under `~/tmp/` and only keep manifests,
  configs, and compact summaries here

## 3) Phase A: no-code tuning (eat the low-risk wins first)

A1) Train time + early stopping:

- grid:
  - `num_epochs`: {50, 100, 200}
  - `early_stopping_patience`: {10, 20, 30}
  - `val_fraction`: {0.1, 0.2}
- goal: improve `anno_lv4 macro_f1 / balanced_accuracy` without harming full-path metrics

A2) MLP capacity + dropout:

- grid:
  - `hidden_sizes`: {[256], [512], [512, 256], [1024, 512]}
  - `dropout_rate`: {0.0, 0.1, 0.2, 0.3}
- goal: better fine-label separation with controlled overfitting

A3) HVG size tradeoff (plus `whole` control points):

- HVG grid: {3000, 5000, 6000, 8000}
- controls: keep 1-2 `whole` points as stability anchors in comparisons

A4) Task weighting validation:

- start from `phmap-like` weights
- probe: `lv4` upweight multipliers {1.5, 2.0, 3.0} with renormalization
- requirement: re-validate on a second dataset or an alternate split

Deliverable after Phase A:

- pick one "best-known stable" config for Phase B that passes the hard constraints

## 4) Phase B: code-change experiments (ordered by expected ROI)

B1) Address class imbalance at `anno_lv4`:

- loss-only round:
  - per-class weighting:
    - `1/sqrt(freq)` (normalized)
    - effective-number weighting (Cui et al.-style)
  - focal loss:
    - `gamma` in {1, 2}
    - `alpha` from the same per-class weights
- sampler-only round:
  - class-balanced sampler in training batches for `anno_lv4`
- combo round:
  - best loss + best sampler

B2) Make hierarchy consistency part of training:

- hierarchical constraint loss:
  - penalize probability mass assigned to invalid child classes per parent
  - `lambda` in {0.1, 0.3, 1.0}
- parent-conditioned decoding:
  - feed parent soft logits/embedding into child head
- guardrail:
  - must not inflate `Unknown` by construction; keep coverage as a primary metric

B3) Improve fine-label representations (metric-learning regularizers):

- supervised contrastive loss on encoder latent:
  - positives: same `anno_lv4` (or same full-path, as a variant)
  - `beta` in {0.05, 0.1, 0.2}
- batching requirement:
  - ensure >= 2 samples per class in batch (may require mild re-sampling)

B4) Improve confidence quality:

- calibration-by-default when validation exists:
  - temperature scaling per level
- optional label smoothing at `anno_lv4`:
  - `epsilon` in {0.05, 0.1}

## 5) Phase C: uncertainty reporting / confidence intervals (metrics)

Two complementary uncertainty sources should be separated:

- evaluation uncertainty (fixed model, resample query cells)
- training stochasticity (retrain across seeds / data resamples)

Recommended reporting:

- for the chosen best config, run N=5 seeds; report mean±std
- additionally, compute stratified bootstrap CIs (e.g. B=1000) over query cells
  for `anno_lv4 macro_f1`, `balanced_accuracy`, and full-path metrics

## 6) Phase D: decision robustness (reduce single-run stochasticity)

Goal: make final predicted labels less sensitive to a single training run.

D1) Remove avoidable non-determinism (must-do hygiene):

- fix seeds for Python/NumPy/PyTorch
- ensure dropout is disabled at inference (unless MC-dropout is explicitly enabled)
- prefer deterministic preprocessing choices (HVG selection, label encoding order)
- when feasible, enable PyTorch deterministic algorithms and record the setting

D2) Single-artifact stabilization (lowest operational overhead):

- EMA (exponential moving average) of weights during training, evaluate with EMA weights
- SWA (stochastic weight averaging) over the last K epochs around the best checkpoint

D3) Snapshot ensemble (no extra training runs):

- save checkpoints from the last K epochs (or the best K epochs by val metric)
- at inference, average per-level probabilities (or logits) across snapshots

D4) Seed ensemble (best robustness, higher cost):

- train M=3 (or 5) models with different seeds on the same data/config
- at inference, average per-level probabilities (or logits) across members
- use ensemble disagreement (variance/entropy) to drive Unknown/abstention policies

Preference order for this cycle:

1) D1 + D2 (if we implement either EMA or SWA cleanly)
2) D1 + D3 (if we want robustness without multi-run training cost)
3) D1 + D4 (if we can afford training/inference cost for maximum robustness)

## 7) Decision log (2026-03-02)

Locked preference for "robust decisions with low resource overhead":

- prioritize **single-model SWA** (or EMA as fallback) to reduce training stochasticity
- keep it as an experiment toggle first; do not change defaults until validated
- prefer CPU-first deployment; single-model artifact is the default operational target
