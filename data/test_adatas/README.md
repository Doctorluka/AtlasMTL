# Local Benchmark Test AnnData

This directory contains local benchmark copies used for atlasmtl development:

- `sampled_adata_10k.h5ad`
  - intended reference dataset
- `sampled_adata_3000.h5ad`
  - intended query dataset

Both files were copied from:

- `/home/data/fhz/project/phmap_package/data/test_adatas/`

Shared properties:

- contain real labels in:
  - `anno_lv1`
  - `anno_lv2`
  - `anno_lv3`
  - `anno_lv4`
- currently contain no coordinate embeddings in `obsm`

Current implication:

- these files are directly usable for label-only benchmarking
- they are not directly compatible with the current default atlasmtl training
  path because `build_model()` currently expects coordinate targets such as
  `X_ref_latent` and optionally `X_umap`

Next implementation step:

- add a no-coordinate training mode for phmap-style benchmark datasets, or
- derive and persist reference/query embeddings before atlasmtl training
