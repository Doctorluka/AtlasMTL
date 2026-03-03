# Preprocessing refactor review plan

Date: `2026-03-03`

This document records the implementation boundary for the first preprocessing
refactor wave before broader benchmark execution.

## Refactor objective

Upgrade the dedicated preprocessing layer so it can support:

- structured counts detection
- `adata.X -> layers["counts"]` promotion when justified
- more flexible Ensembl canonicalization
- reference-derived HVG selection after canonicalization
- split-ready prepared assets
- stronger experiment traceability

## Locked preprocessing order

1. load input AnnData
2. detect matrix semantics
3. validate or materialize `layers["counts"]`
4. canonicalize gene IDs to versionless Ensembl
5. derive reference feature panel (`whole` or `hvg`)
6. align query to the feature panel
7. materialize split assets
8. run benchmark on prepared assets only

## First-wave required code changes

- enrich `PreprocessConfig`
- enrich `PreprocessReport`
- refactor `matrix_semantics.py`
- refactor `gene_ids.py`
- refactor `pipeline.py`
- add dedicated split helpers
- keep benchmark runner consuming prepared assets instead of owning split logic

## Required first-wave evidence

- PH-Map path works with existing `layers["counts"]`
- DISCO path works by confirming count-like `adata.X` and promoting it
- Ensembl canonicalization supports:
  - versionless Ensembl
  - versioned Ensembl
  - symbol + existing Ensembl column
  - symbol-only mapping-table conversion
- HVG selection remains anchored to `layer="counts"`
- preprocessing metadata is exported into run outputs

## Documentation and record requirement

Each first-wave run must leave:

- an execution report
- an experiment record
- a discussion note about failures, warnings, and next-round changes
