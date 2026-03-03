# HLCA pilot protocol

Date: `2026-03-03`

## Objective

Establish the first reference-specific benchmark protocol for `HLCA_Core`,
keeping its log-normalized `adata.X` and `layers["counts"]` contract explicit.

## Dataset contract

- Reference path:
  `/home/data/fhz/project/phmap_package/data/real_test/HLCA/hlca_clean.h5ad`
- Primary split field: `donor_id`
- Current default domain key: `donor_id`
- Primary fine target: `ann_level_5`
- Coarse target for optional single-level comparison: `ann_level_3`
- Hierarchical targets:
  `ann_level_1`, `ann_level_2`, `ann_level_3`, `ann_level_4`, `ann_level_5`
- Counts source: `layers["counts"]`
- `adata.X` semantics: log-normalized expression
- `ann_finest_level`: not used

## First-pass scenario

- Scenario class: `reference_heldout`
- Recommended first manifest:
  `HLCA_Core__ann_level_5__group_split_v1.yaml`
- Quantitative claim boundary:
  - this scenario may be used for formal accuracy / calibration / coverage
    reporting once the split is locked

## Fairness constraints

- one shared heldout truth pool for all methods
- one shared target label column: `ann_level_5`
- one shared training-size point per run
- one shared heldout evaluation-size point per run
- same counts contract declaration for all methods

## Dataset-specific tuning allowed

- domain key choice among `donor_id`, `Subject group`, `GSE_id`
- dataset-specific feasible size ceiling after split
- HLCA-specific canonicalization note for query-side mixed IDs

## Planned follow-up scenario

- Scenario class: `external_query_validation`
- Query path:
  `/home/data/fhz/project/phmap_package/data/real_test/query_data/hlca_query_GSE302339.h5ad`
- Intended outputs:
  - Sankey against `Gold` only as visualization unless special approval is recorded
  - marker-based dotplot / heatmap
  - confidence / Unknown distribution

## Note for later rounds

- HLCA is not part of the first `5k/1k` flow-through pilot pair
- keep this dossier as the next reference-specific expansion target after the
  PH-Map and DISCO first-wave preprocessing path is validated
