# AtlasMTL Methods Outline

This outline fixes the paper-facing Methods structure for the AtlasMTL tool
paper. It is intended to be checked section by section before prose drafting.

## 1. Method Overview

Purpose:

- define AtlasMTL as a multi-level `sc -> sc reference mapping` framework
- anchor the paper around reliable label transfer rather than integrated
  embedding leadership

Must-cover points:

- reference input and query input are both `AnnData`
- the framework transfers multi-level cell labels from a labeled reference to
  unlabeled query cells
- the primary outputs are predicted labels, confidence signals, abstention /
  Unknown behavior, and traceable prediction metadata
- the method is designed for accurate, reliable, hierarchically consistent, and
  traceable annotation

Evidence anchor:

- `documents/design/research_positioning.md`

## 2. Data Contract And Preprocessing

Purpose:

- define the reproducible input contract used by AtlasMTL and by the formal
  benchmark pipeline

Must-cover points:

- `AnnData in, AnnData out`
- versionless Ensembl IDs as the canonical internal feature namespace
- `adata.layers["counts"]` as the raw-count contract
- explicit handling of symbol-only inputs through a documented preprocessing
  step rather than implicit acceptance
- feature panel locking and reference-query feature alignment
- default training input transform: `input_transform="binary"`
- formal benchmarking kept the preprocessing contract fixed across methods and
  refresh rounds

Recommended subsection order:

1. input matrices and counts-layer contract
2. gene identifier normalization
3. feature panel / HVG selection
4. binary input transform
5. reference-query feature alignment

Evidence anchor:

- `AGENTS.md`
- `documents/design/api_contract.md`

## 3. Multi-Task Architecture

Purpose:

- describe the core AtlasMTL model family in framework language

Must-cover points:

- hard parameter-sharing multi-task architecture
- one shared backbone for feature extraction
- one task-specific classification head per annotation level
- weighted multi-task training objective across levels
- do not imply that a single benchmark-default non-uniform task-weight
  schedule has already been fixed for all multi-level datasets
- architecture serves hierarchical label transfer, not a primary integration
  benchmark claim

Safe wording:

- keep the main text at the level of architecture family and task structure
- move exact layer widths / hidden sizes into implementation details or
  supplement unless they are needed in the main text

## 4. Training Procedure And Software Defaults

Purpose:

- define how AtlasMTL models are trained and which defaults are paper-safe to
  state

Must-cover points:

- training begins from preprocessed reference data
- validation-based early stopping is used
- current software default training path is:
  - `input_transform="binary"`
  - `optimizer_name="adamw"`
  - `weight_decay=5e-5`
  - `scheduler_name=None`
  - `reference_storage="external"`
- `2026-03-07` parameter-lock round fixed the stable training skeleton
- `2026-03-09` low-cost optimization round promoted the software optimizer
  default
- software default and manuscript-grade formal table rows are separate records

Mandatory caveat:

- do not write that the formal manuscript-grade AtlasMTL rows were replaced by
  the refreshed default

Evidence anchor:

- `documents/paper/methods/2026-03-09_atlasmtl_training_defaults_summary.md`

## 5. Prediction, Confidence, Abstention, Hierarchy, And Optional KNN Refinement

Purpose:

- describe AtlasMTL as an annotation workflow rather than just a classifier

Must-cover points:

- simultaneous multi-level predictions on query cells
- per-level probability outputs and decoded labels
- confidence / margin style uncertainty signals
- abstention / Unknown behavior as part of the method contract
- hierarchy consistency as part of interpretation and quality control
- optional KNN-assisted refinement, if mentioned, should be framed as a
  supplementary extension rather than a required main-path component

Recommended subsection order:

1. direct model prediction
2. confidence outputs
3. abstention / Unknown handling
4. hierarchy-aware interpretation
5. optional KNN refinement and audit fields

## 6. Model Artifacts And Reproducibility

Purpose:

- emphasize that AtlasMTL is a reproducible tool workflow, not only a network

Must-cover points:

- exported artifacts form a portable trained model package
- preferred layout:
  - `model.pth`
  - `model_metadata.pkl`
  - `model_reference.pkl`
  - `model_manifest.json`
- artifacts record train configuration, feature panel, label metadata, and
  reference linkage
- `build -> save -> load -> predict` is a supported reproducible workflow

## 7. Benchmark Design And Evaluation

Purpose:

- define the evaluation logic for a bioinformatics tool paper

Must-cover points:

- primary task is `sc -> sc reference mapping`
- comparator family stays within the same task class
- primary evaluation metrics:
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
- resource metrics:
  - training time
  - prediction time
  - peak RSS
  - peak GPU memory
- CPU/GPU fairness policy and degraded-runtime flags are part of the benchmark
  contract

Mandatory caveat:

- low-cost optimization established the software default
- fifth-round formal refresh did not justify replacing retained third-wave
  manuscript-grade AtlasMTL rows
- sixth-round multi-level benchmarking should be framed as a capability study,
  not as a replacement for retained single-level formal rows

## 8. Implementation Details And Paper Boundary Notes

Purpose:

- prevent later prose drift between software state and manuscript evidence

Must-cover points:

- separate "current software default" from "formal comparison-table rows"
- keep the current software default in Methods
- keep retained third-wave AtlasMTL rows as manuscript-grade formal comparison
  rows
- treat the fifth-round formal refresh as a methodological clarification, not a
  formal main-table upgrade

## Section-By-Section Review Checklist

Use this checklist before locking prose:

1. Every section states framework behavior, not dataset-specific story, unless
   the subsection is explicitly benchmark-specific.
2. Every claim about defaults is labeled as either software default or formal
   manuscript record.
3. No section frames AtlasMTL primarily as an integrated embedding method.
4. Prediction behavior includes confidence, abstention, and optional KNN
   refinement, with KNN treated as optional rather than central.
5. Reproducibility language includes artifact layout and metadata capture.
6. Benchmark language keeps the formal retained rows and refreshed software
   defaults separate.
