# 2026-03-02: KNN external space (X_scANVI) + data IO optimizations

## Summary

This update adds a reproducible way to evaluate AtlasMTL KNN correction using an external, precomputed embedding space stored in `AnnData.obsm` (e.g. `X_scANVI`). It also improves feature alignment and matrix extraction performance for sparse inputs.

## Changes

### KNN: explicit external-space support

- `build_model(..., knn_reference_obsm_key=..., knn_space=...)` can now store a reference embedding from `adata.obsm` **for KNN only** (no coordinate regression head is trained).
- `predict(..., knn_query_obsm_key=..., knn_space=...)` can now use `adata.obsm[knn_query_obsm_key]` as the KNN query space.
- `resolve_knn_space(..., preferred=...)` now supports a strict preferred-space mode so benchmark manifests can guarantee which space was used.
- Benchmark runner (`benchmark/pipelines/run_benchmark.py`) supports manifest keys:
  - `train.knn_reference_obsm_key`, `train.knn_space`
  - `predict.knn_query_obsm_key`, `predict.knn_space`

### KNN: internal latent fallback (no coord heads)

- Training stores `reference_data.coords["X_ref_latent_internal"]` (internal encoder latent) so KNN correction can be evaluated even when no `coord_targets` are configured.

### Performance: sparse alignment + extraction

- Sparse feature alignment no longer performs expensive CSR structural assignment when reindexing to the feature panel.
- Matrix extraction no longer densifies the full `adata.X` when only a subset of columns (training genes) is needed.

### Metrics stability

- Balanced-accuracy computation now avoids sklearn warnings when `pred` contains labels not present in `true` (e.g. `Unknown` abstention).

## Local smoke run (not paper-final)

KNN ablation using `obsm["X_scANVI"]` on the local 10k reference / 3k query pair ran successfully on CPU. Outputs are under:

- `~/tmp/atlasmtl_knn_scanvi_eval_20260302/metrics.json`

Do not treat this as a paper-final result; replicate across datasets and devices.

