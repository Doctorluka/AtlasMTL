# Execution Plan

Locked decisions for this run:

- authoritative gene mapping resource:
  `atlasmtl/resources/gene_id/biomart_human_mouse_rat.tsv.gz`
- input namespace:
  `var_names_type="symbol"`
- species:
  `human`
- raw counts source:
  `layers["counts"]`
- single-level comparator target:
  `anno_lv4`
- multi-level AtlasMTL labels:
  `anno_lv1`, `anno_lv2`, `anno_lv3`, `anno_lv4`
- feature space:
  `whole`
- unmapped policy:
  `drop`
- duplicate policy:
  `sum`
- KNN correction for this run:
  `off`

Run phases:

1. Audit raw datasets and record counts/gene-ID preprocessing facts.
2. Preprocess reference and query with the bundled BioMart table.
3. Train a single-level CellTypist comparator model on the preprocessed
   reference for `anno_lv4`.
4. Run the single-level comparator benchmark on preprocessed inputs.
5. Run the multi-level AtlasMTL evaluation with hierarchy enforcement.
6. Summarize results and record exclusions or failures.
