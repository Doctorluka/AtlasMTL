# Overall Assessment

What is already closed in this run:

- raw-data audit and preprocessing facts are recorded
- bundled BioMart-based `symbol -> Ensembl` preprocessing was executed on the
  real sampled datasets
- a single-level benchmark was completed for:
  - `atlasmtl`
  - `reference_knn`
  - `celltypist`
  - `scanvi`
  - `singler`
  - `symphony`
  - `azimuth`
- a multi-level AtlasMTL run on `anno_lv1` to `anno_lv4` completed with
  hierarchy enforcement

Current project-level conclusion from this run:

- the current AtlasMTL architecture is sufficient to run:
  - single-level shared-target benchmarking
  - multi-level label transfer with hierarchy-aware evaluation
- the current sampled datasets are sufficient for:
  - label-transfer benchmarking
  - hierarchy consistency benchmarking
- the current sampled datasets are not sufficient for:
  - KNN correction evaluation
  - coordinate-regression or coordinate-metric evaluation

Recommended next action after this run:

1. preserve `all_methods_final_v2` as the canonical sampled single-level bundle
2. if KNN correction needs formal evaluation, prepare reference/query latent or
   coordinate targets
3. if a strict native Azimuth result is required, rerun with a setup that
   avoids fallback and record the backend explicitly
