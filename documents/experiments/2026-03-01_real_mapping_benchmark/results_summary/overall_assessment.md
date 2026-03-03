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
  - `seurat_anchor_transfer`
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

Runtime/resource recording status:

- yes, runtime fields were recorded for all completed methods in the benchmark
  `metrics.json`
- AtlasMTL recorded both elapsed time and peak RSS on this run
- most comparator wrappers currently record elapsed time but still leave
  `process_peak_rss_gb` as `null`

Recommended next action after this run:

1. preserve `all_methods_final_v2` as the canonical sampled single-level bundle
2. extend comparator wrappers toward fully standardized average/peak memory and
   CPU-usage reporting, then rerun the formal resource table export
3. benchmark AtlasMTL CPU and GPU modes as separate formal variants
4. if KNN correction needs formal evaluation, prepare reference/query latent or
   coordinate targets
5. keep the Seurat comparator recorded under the explicit
   `seurat_anchor_transfer` name and backend
