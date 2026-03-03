# PH-Map pilot protocol

Date: `2026-03-03`

## Objective

Establish the first reference-specific benchmark protocol for
`PHMap_Lung_Full_v43_light` without coupling it to other reference datasets.

## Dataset contract

- Reference path:
  `/home/data/fhz/project/phmap_package/data/real_test/phmap/phmap_clean.h5ad`
- Primary split field: `sample`
- Locked first-wave domain key: `sample`
- Primary fine target: `anno_lv4`
- Coarse target for optional single-level comparison: `anno_lv2`
- Hierarchical targets: `anno_lv1`, `anno_lv2`, `anno_lv3`, `anno_lv4`
- Counts source: `layers["counts"]`
- `var_names_type`: symbol-like

## First-pass scenario

- Scenario class: `reference_heldout`
- Recommended first manifest:
  `PHMap_Lung_Full_v43_light__anno_lv4__group_split_v1.yaml`
- Locked first-wave subset sizes:
  - reference build subset: `5k`
  - prediction subset: `1k`
- Quantitative claim boundary:
  - this scenario may be used for formal accuracy / calibration / coverage
    reporting once the split is locked

## First-wave preprocessing decision

- counts detection is still run first, even though `layers["counts"]` already exists
- this pilot exercises the standard `layers["counts"]` path
- gene ID canonicalization runs immediately after counts validation
- prepared subsets should be written under `~/tmp/atlasmtl_benchmarks/...`
- repo-side follow-up must include a report, summary, and discussion note

## Fairness constraints

- one shared heldout truth pool for all methods
- one shared target label column: `anno_lv4`
- one shared training-size point per run
- one shared heldout evaluation-size point per run
- same counts contract declaration for all methods

## Dataset-specific tuning allowed

- dataset-specific feasible size ceiling after split
- PH-Map-specific gene canonicalization note

## Planned follow-up scenario

- Scenario class: `external_query_validation`
- Query path:
  `/home/data/fhz/project/phmap_package/data/real_test/query_data/query_PH.h5ad`
- Intended outputs:
  - Sankey if a usable author-side grouping is approved
  - marker-based dotplot / heatmap
  - confidence / Unknown distribution
