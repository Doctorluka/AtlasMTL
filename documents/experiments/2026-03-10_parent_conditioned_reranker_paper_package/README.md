# Parent-Conditioned Reranker Paper Package (`2026-03-10`)

Status: completed

This package consolidates the publication-oriented outputs from:

- `2026-03-09_phmap_study_split_validation`
- `2026-03-10_hlca_study_split_refinement_validation`

Purpose:

- provide unified main-figure and supplementary-table exports for the current
  paper chapter
- keep PH-Map and HLCA evidence aligned to a single panel-oriented schema
- provide a concise design summary and current-claims summary for expert review

Current framing:

- PH-Map is the finalized positive hard-case dataset
- HLCA is the second deep-hierarchy validation dataset, currently with:
  - positive support for dataset-specific weighting selection
  - mixed first-pass evidence for auto reranking

Key outputs are generated under:

- `results_summary/paper_package/`

Generator script:

- `scripts/export_parent_conditioned_reranker_paper_package.py`

Current package outputs include:

- unified main-figure tables
- unified supplementary tables
- chapter design summary
- current-results summary
- Chinese discussion note for expert review
