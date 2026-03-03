# Decision Robustness Discussion (2026-03-02)

## Goal

Make AtlasMTL final decisions **more robust** with **low resource overhead**:

- reduce sensitivity to single-run randomness (training stochasticity)
- keep inference cheap (CPU-first deployment preference)
- keep or improve performance (esp. `anno_lv4`) without harming hierarchy/coverage

## Current baseline context

- default inference remains `knn_correction="off"` (KNN not prioritized)
- model quality headroom is mainly at `anno_lv4`
- validation + early stopping exist; temperature scaling calibration exists

## Uncertainty sources (separate them)

1) evaluation uncertainty: fixed model, resample query cells (bootstrap metrics)
2) training stochasticity: different seeds / resampled training data

For the goal "decision robustness", we primarily target (2).

## Candidate robustness schemes (tradeoffs)

### A) Single-model stabilization (preferred)

- SWA (stochastic weight averaging) or EMA during training
- output artifact remains **one model**
- inference remains **one forward pass**

Why preferred: best fit to "low resource + high efficiency + stable decisions".

### B) Snapshot ensemble (same training run, slower inference)

- save K checkpoints, average probabilities/logits at inference
- no extra training runs, but inference cost multiplies by K

Use as optional if SWA/EMA underperforms and we still cannot afford multi-run training.

### C) Seed ensemble (best robustness, highest cost)

- train M models with different seeds, average at inference
- best stability and often best calibration, but higher training + storage + inference

Keep as non-default baseline for papers / high-stakes settings if resources allow.

## Decision (locked for this cycle)

- Primary robustness method to prioritize: **SWA single-model**.
- Default behavior: **do not change defaults yet**; run as an experiment toggle and decide after results.
- Deployment preference: **CPU-first**; keep single-model artifact for easier transfer/rollout.

## Cross-device transfer note

- Single-model artifacts are operationally simpler to move across CPU/GPU.
- Exact bitwise identical predictions across devices are not guaranteed due to floating-point/ops differences.
- Single-model + SWA/EMA tends to reduce boundary sensitivity, making cross-device label flips less likely.

## Acceptance criteria (for adopting SWA)

- across N=5 seeds: `anno_lv4 macro_f1` and `balanced_accuracy` std decreases ≥ 25%
- means do not regress, and `anno_lv1..lv3` stay within noise
- full-path coverage does not collapse (guardrail)
- inference cost stays ~unchanged vs baseline

