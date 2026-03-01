# Built-in gene ID resources

This directory stores atlasmtl-packaged gene-identifier resources used by
preprocessing utilities and protocol documentation.

Current bundled resource:

- `biomart_human_mouse_rat.tsv.gz`
  - source table:
    `/home/data/public_data/database/bioMart/GRCh38_Human_Rat_Mouse.txt`
  - rewritten with atlasmtl-oriented column names
  - canonical columns:
    - `human_ensembl_gene_id`
    - `human_gene_symbol`
    - `human_gene_description`
    - `mouse_ensembl_gene_id`
    - `mouse_gene_symbol`
    - `rat_ensembl_gene_id`
    - `rat_gene_symbol`

The file is intended as a bundled mapping baseline for preprocessing. It does
not replace explicit experiment metadata about species, source namespace, or
duplicate/unmapped-gene handling.
