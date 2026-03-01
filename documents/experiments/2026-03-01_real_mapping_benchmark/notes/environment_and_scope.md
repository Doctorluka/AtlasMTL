# Environment And Scope Notes

Runtime environment:

- Python: `/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python`
- native `Azimuth` / `Seurat v5` R library:
  `/home/data/fhz/seurat_v5`
- repo-local comparator R library:
  `/home/data/fhz/project/phmap_package/atlasmtl/.r_libs`

Runtime storage:

- repo-tracked materials:
  `documents/experiments/2026-03-01_real_mapping_benchmark/`
- runtime artifacts:
  `~/tmp/atlasmtl_real_mapping_benchmark_20260301/`

Run exclusions:

- KNN correction is disabled for this run because the sampled reference/query
  AnnData files do not provide coordinate targets in `obsm`.

Gene-ID policy for this run:

- input namespace is `symbol`
- canonical internal namespace is versionless Ensembl
- authoritative mapping source is the bundled BioMart table at
  `atlasmtl/resources/gene_id/biomart_human_mouse_rat.tsv.gz`
