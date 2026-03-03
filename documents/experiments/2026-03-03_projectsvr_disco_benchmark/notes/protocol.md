# DISCO pilot protocol

Date: `2026-03-03`

## Objective

Establish the first ProjectSVR pilot benchmark protocol using
`DISCO_hPBMCs`, while explicitly validating the case where counts live in
`adata.X` and `layers["counts"]` is absent.

## Dataset contract

- Reference path:
  `/home/data/fhz/project/phmap_package/data/real_test/ProjectSVR/reference_atlas/DISCO_hPBMCs.h5ad`
- Primary split field: `sample`
- Locked first-wave domain key: `sample`
- Primary fine target: `cell_subtype`
- Coarse target for optional single-level comparison: `cell_type`
- Counts source: count-like `adata.X`
- `layers["counts"]`: absent by design for this pilot
- `var_names_type`: symbol-like

## First-pass scenario

- Scenario class: `reference_heldout`
- Recommended first manifest:
  `DISCO_hPBMCs__cell_subtype__group_split_v1.yaml`
- Locked first-wave subset sizes:
  - reference build subset: `5k`
  - prediction subset: `1k`
- Quantitative claim boundary:
  - this scenario may be used for formal accuracy / calibration / coverage
    reporting once the split is locked

## First-wave preprocessing decision

- counts detection is the first required test for this pilot
- only if `adata.X` is classified as `counts_confirmed` may preprocessing copy
  `adata.X` into `layers["counts"]`
- after that promotion step, the dataset enters the same standard preprocessing
  path as other pilots
- gene ID canonicalization runs immediately after counts validation/promotion
- prepared subsets should be written under `~/tmp/atlasmtl_benchmarks/...`
- repo-side follow-up must include a report, summary, and discussion note

## Fairness constraints

- one shared heldout truth pool for all methods
- one shared target label column: `cell_subtype`
- one shared training-size point per run
- one shared heldout evaluation-size point per run
- same explicit matrix-semantics declaration for all methods

## Dataset-specific tuning allowed

- explicit preprocessing note that `input_matrix_type` is count-like in `adata.X`
- dataset-specific feasible size ceiling after split
- optional secondary coarse-level run on `cell_type`

## Planned follow-up scenario

- Scenario class: `external_query_validation`
- Query path:
  `/home/data/fhz/project/phmap_package/data/real_test/ProjectSVR/query_data/pbmc_query.h5ad`
- Intended outputs:
  - Sankey against author-side `cell_type` / `cell_subtype`
  - marker-based dotplot / heatmap
  - confidence / Unknown distribution
