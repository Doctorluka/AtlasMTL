# AtlasMTL Methods Open Items

This file records the remaining items that should be resolved or checked before
locking the final manuscript Methods prose.

## 1. Architecture-Level Specificity

Current status:

- the framework shape is fixed
- the main-text / supplement split is now defined

Open decision:

- whether later reviewer feedback requires exact hidden-layer widths in the
  main text

Default recommendation:

- keep main text at the architecture-family level
- keep exact widths and benchmark-facing default grids in supplement or an
  implementation-details subsection

## 2. Loss Weighting Description

Current status:

- the framework-level weighted multi-task formulation is stable
- the benchmark-default weight setting is not yet stable for the multi-level
  round

Open check:

- verify the most paper-safe description of task weighting from the current
  training code and completed experiment record before final prose is written

Reason:

- the sixth-round `v1` multi-level benchmark used uniform weights
- follow-up probes showed that fine-level upweight can materially help, but the
  best schedule appears dataset-sensitive rather than globally fixed
- this is easy to overstate if the prose is borrowed too directly from the
  PH-Map atlas-specific writeup

Default recommendation:

- main Methods: describe AtlasMTL as supporting weighted multi-task learning
- do not imply that a single benchmark-default non-uniform weight schedule has
  already been established across all multi-level datasets

## 3. Abstention, Hierarchy, And Optional KNN Wording

Current status:

- abstention and hierarchy consistency are central to AtlasMTL positioning
- KNN is currently an optional extension, not part of the active benchmark
  default path
- a main-text / supplement split has been defined

Open check:

- verify whether any reviewer-facing draft needs more implementation detail for:
  - abstention trigger description
  - hierarchy consistency handling
  - optional KNN rescue / KNN audit description, if KNN is mentioned at all

Default recommendation:

- keep the main Methods focused on abstention behavior and hierarchy-aware
  consistency
- treat KNN as optional or supplementary unless a later round restores it to
  the main evidence path
- move threshold-level or audit-column-level detail to supplement if it makes
  the prose too operational

## 4. Artifact-Level Detail

Current status:

- artifact layout is stable enough to describe
- main-text / supplement split is now defined

Open check:

- verify whether explicit artifact filenames are required outside supplement
  and repository-linked methods docs

Default recommendation:

- mention the existence of a portable bundle in main text
- keep exact filenames available in supplement or repository-facing methods docs

## 5. Benchmark Dataset Paragraphing

Current status:

- benchmark task family and evaluation logic are stable

Open decision:

- whether the main Methods should name all benchmark datasets in one paragraph,
  or split dataset inventory into a separate benchmark protocol / supplement
  paragraph

Default recommendation:

- main Methods: describe benchmark design and task family
- supplement or benchmark subsection: enumerate dataset roster and split design

## 6. Formal-Refresh Disclosure Placement

Current status:

- the decision itself is fixed
- the default placement is now defined

Open decision:

- whether an additional one-sentence reminder is needed in the Results section
  to prevent readers from conflating software defaults and retained formal rows

Default recommendation:

- mention software default in the training subsection
- mention retained formal rows in the benchmark / reporting subsection
- avoid putting the full fifth-round explanation in the method overview

## 7. Implementation Detail Verification Pass

Current status:

- the structure is ready for prose drafting

Required pre-draft check:

- before writing final manuscript prose, verify the following directly against
  code and docs:
  - exact description of feature alignment
  - exact description of exported artifact contents
  - exact description of prediction metadata fields safe for paper mention
  - exact description of benchmark fairness metadata terminology

## 8. What Is Already Closed

These items should be treated as fixed and should not be reopened during prose
drafting unless the code or experiment record changes:

- AtlasMTL paper positioning is tool/framework-first, not integration-first
- current software default is `AdamW + weight_decay=5e-5`
- `ReduceLROnPlateau` is not a default
- retained third-wave AtlasMTL rows remain the manuscript-grade formal
  comparison rows
- fifth-round formal refresh is a non-replacement result
- sixth-round multi-level benchmark is a capability round, not a replacement
  for the retained single-level formal comparison rows
- current sixth-round evidence supports hierarchy consistency strongly, but
  does not yet fix a universal non-uniform task-weight schedule
