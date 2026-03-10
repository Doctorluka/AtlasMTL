# AtlasMTL Methods Fact Sheet

This file lists paper-safe facts that are already supported by code, design
documents, or completed experiment records. It is intended to prevent writing
drift when drafting the Methods section.

## A. Positioning Facts

- AtlasMTL is positioned as a multi-level `sc -> sc reference mapping` method.
- The primary research claim is reliable label transfer under uncertainty.
- The main paper claim is not that AtlasMTL is the strongest integrated latent
  embedding method.
- Primary method outputs include predicted labels, confidence-related signals,
  abstention behavior, and traceable run metadata.

Primary sources:

- `documents/design/research_positioning.md`

## B. Input And Preprocessing Facts

- AtlasMTL uses `AnnData` as the primary input/output container.
- Versionless Ensembl IDs are the canonical internal gene namespace for formal
  training and prediction workflows.
- `adata.layers["counts"]` is the expected raw-count layer contract for formal
  preprocessing.
- If an input arrives with symbols only, a documented preprocessing step is
  required; symbol-only alignment should not be described as an implicit formal
  input path.
- Feature alignment is performed against the training feature panel, with
  missing features padded during alignment.
- The current default training input transform is `input_transform="binary"`.

Primary sources:

- `AGENTS.md`
- `documents/design/api_contract.md`

## C. Architecture Facts

- AtlasMTL uses a hard parameter-sharing multi-task architecture.
- A shared backbone is paired with task-specific output heads.
- Each output head corresponds to one annotation level.
- Training minimizes a weighted objective across multiple annotation levels.
- The framework-level weighted formulation is stable, but a universal
  benchmark-default non-uniform task-weight schedule has not yet been fixed
  across all multi-level datasets.
- The architecture should be described as supporting hierarchical label
  transfer rather than as a primary integrated representation objective.

Primary sources:

- code architecture in the training pipeline
- `documents/design/research_positioning.md`

## D. Training-Default Facts

- The benchmark-facing training skeleton was locked in the
  `2026-03-07_atlasmtl_param_lock_benchmark` round.
- The current software default optimizer path was promoted in the
  `2026-03-09_atlasmtl_low_cost_optimization` round.
- The current software default training path is:
  - `input_transform="binary"`
  - `optimizer_name="adamw"`
  - `weight_decay=5e-5`
  - `scheduler_name=None`
  - `reference_storage="external"`
- `ReduceLROnPlateau` is not part of the default configuration.
- The default-promotion evidence came primarily from non-degraded Stage B GPU
  confirmation.

Primary sources:

- `documents/paper/methods/2026-03-09_atlasmtl_training_defaults_summary.md`
- `documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/results_summary/final_default_decision.md`

## E. Formal-Record Facts

- The fifth-round formal refresh reran only AtlasMTL under the promoted
  software default.
- The fifth-round formal refresh did not justify replacing the retained
  third-wave manuscript-grade AtlasMTL baseline rows.
- Across the `16` main-panel rows, mean `delta_macro_f1 = -0.005250`.
- GPU headline improvements in the fifth-round formal refresh were `4/8`.
- A substantial regression occurred at
  `DISCO_hPBMCs / gpu / predict_100000_10000` with `delta_macro_f1 = -0.054224`.
- Therefore, retained third-wave AtlasMTL rows remain the manuscript-grade
  formal comparison rows.
- The promoted software default still remains acceptable as the software
  default, even though it was not promoted into manuscript-table replacement.

Primary sources:

- `documents/experiments/2026-03-09_formal_atlasmtl_refresh/results_summary/formal_refresh_decision.md`
- `documents/experiments/2026-03-09_formal_atlasmtl_refresh/results_summary/formal_refresh_results.csv`

## F. Prediction-Workflow Facts

- AtlasMTL predicts multiple annotation levels for each query cell.
- Prediction outputs include decoded labels and probability-derived confidence
  information.
- Abstention / Unknown behavior is part of the evaluation and reporting
  contract.
- Hierarchy consistency should be treated as a method-level property rather
  than only a visualization detail.
- Optional KNN-assisted refinement exists in the framework, but it is not part
  of the current main benchmark path and should not be described as a required
  active component of the present paper evidence line.

Primary sources:

- `documents/design/research_positioning.md`
- prediction metadata and benchmark metric contracts

## G. Reproducibility Facts

- AtlasMTL exports portable trained-model artifacts rather than notebook-only
  state.
- The preferred artifact layout is:
  - `model.pth`
  - `model_metadata.pkl`
  - `model_reference.pkl`
  - `model_manifest.json`
- Artifact metadata should be described as recording train configuration,
  feature metadata, and reference linkage.
- The tool supports a reproducible `build -> save -> load -> predict` workflow.

Primary sources:

- `AGENTS.md`
- artifact / serialization code paths

## H. Benchmark-Method Facts

- AtlasMTL benchmark evaluation is centered on `sc -> sc reference mapping`.
- The primary benchmark metrics should be described as label quality and
  reliability metrics, including:
  - `accuracy`
  - `macro_f1`
  - `balanced_accuracy`
  - `coverage`
  - `reject_rate`
  - `covered_accuracy`
  - `risk`
  - `ece`
  - `brier`
  - `aurc`
- Resource benchmarking includes elapsed time, peak RSS, and peak GPU memory.
- CPU/GPU fairness policy and degraded-runtime flags are part of the formal
  benchmark contract.

Primary sources:

- `documents/design/research_positioning.md`
- benchmark protocol and experiment dossiers

## I. Paper-Safe Wording Rules

- Safe: "current software default"
- Safe: "retained third-wave manuscript-grade AtlasMTL baseline rows"
- Safe: "the fifth-round formal refresh did not justify manuscript-table
  replacement"
- Safe: "the sixth-round multi-level benchmark is a capability round"
- Unsafe without qualification: "the formal benchmark now uses the new
  default"
- Unsafe without qualification: "AtlasMTL formal main-table rows were updated
  to AdamW + wd=5e-5"
- Unsafe without qualification: "AtlasMTL now uses a fixed optimized
  multi-level task-weight schedule"
- Unsafe without qualification: "KNN refinement is part of the current active
  benchmark default"
- Unsafe framing: "AtlasMTL is primarily an integrated embedding method"
