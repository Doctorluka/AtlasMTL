# Raw Data Audit

Datasets:

- reference: `data/test_adatas/sampled_adata_10k.h5ad`
- query: `data/test_adatas/sampled_adata_3000.h5ad`

Observed structure:

- reference shape: `10000 x 21977`
- query shape: `3000 x 21977`
- shared label columns:
  - `anno_lv1`
  - `anno_lv2`
  - `anno_lv3`
  - `anno_lv4`
- shared batch/domain column:
  - `sample`
- `obsm`: empty in both files
- `layers["counts"]`: present in both files

Matrix semantics:

- `adata.X` is not count-like in either file
- `adata.layers["counts"]` is count-like in both files
- this run therefore treats `layers["counts"]` as the authoritative raw-count
  input

Gene-ID preprocessing:

- raw `var_names` are gene symbols
- authoritative mapping resource:
  `atlasmtl/resources/gene_id/biomart_human_mouse_rat.tsv.gz`
- mapping mode:
  `symbol -> versionless Ensembl`

Canonicalization outcome:

- reference:
  - input genes: `21977`
  - canonical genes kept: `21510`
  - unmapped genes dropped: `467`
  - duplicate canonical IDs collapsed: `0`
- query:
  - input genes: `21977`
  - canonical genes kept: `21510`
  - unmapped genes dropped: `467`
  - duplicate canonical IDs collapsed: `0`

Runtime artifact paths:

- preprocessing audit JSON:
  `~/tmp/atlasmtl_real_mapping_benchmark_20260301/prepared/preprocessing_audit.json`
- feature panel:
  `~/tmp/atlasmtl_real_mapping_benchmark_20260301/prepared/feature_panel.json`
- preprocessed reference:
  `~/tmp/atlasmtl_real_mapping_benchmark_20260301/prepared/reference_preprocessed.h5ad`
- preprocessed query:
  `~/tmp/atlasmtl_real_mapping_benchmark_20260301/prepared/query_preprocessed.h5ad`
