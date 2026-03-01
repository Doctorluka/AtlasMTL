# Overall Assessment

What is already closed in this run:

- raw-data audit and preprocessing facts are recorded
- bundled BioMart-based `symbol -> Ensembl` preprocessing was executed on the
  real sampled datasets
- a single-level benchmark was completed for:
  - `atlasmtl`
  - `reference_knn`
  - `scanvi`
- exported prediction files were produced for:
  - `singler`
  - `symphony`
- a multi-level AtlasMTL run on `anno_lv1` to `anno_lv4` completed with
  hierarchy enforcement

What remains incomplete:

- `azimuth` has not closed in the combined R-method benchmark invocation yet
- `celltypist` model training is still unresolved for this sampled real-data
  run

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

1. close the pending `azimuth` and `celltypist` comparator paths or explicitly
   defer them
2. if KNN correction needs formal evaluation, prepare reference/query latent or
   coordinate targets
3. once comparator closure is stable, rerun the single-level benchmark as one
   fully synchronized benchmark bundle
