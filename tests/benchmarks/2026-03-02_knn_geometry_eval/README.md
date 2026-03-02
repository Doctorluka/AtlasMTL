# KNN geometry evaluation (A vs B)

This folder documents a focused benchmark that re-evaluates whether AtlasMTL's
KNN correction is helpful **under the real deployment assumption**:

- query data may only provide an expression matrix and counts layer
- query data should not be assumed to have `obsm["X_scANVI"]`

## What is compared

- **A) predicted coordinate head** (`coord_targets={"scanvi":"X_scANVI"}`), then
  KNN in predicted scanvi space.
- **B) internal latent** (no coord heads), then KNN in `latent_internal` space.

Each geometry mode is tested with:

- `knn_off`
- `knn_lowconf`
- `knn_all`

## Fairness constraint

The main benchmark uses a query `.h5ad` derived from
`data/test_adatas/knn/query_3k.h5ad` by stripping all `obsm/obsp`, ensuring that
all variants are evaluated on the **same cells** without access to embeddings.

## How to run

Run outputs must go to a private location under your home directory, e.g.
`~/tmp/...`.

Example:

`NUMBA_CACHE_DIR=/tmp/numba_cache PYTHONPATH=/home/data/fhz/project/phmap_package/atlasmtl /home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python documents/experiments/2026-03-01_real_mapping_benchmark/ablation_optimization/scripts/run_atlasmtl_knn_geometry_eval.py --dataset-manifest documents/experiments/2026-03-01_real_mapping_benchmark/ablation_optimization/manifests/atlasmtl_knn_geometry_eval_base.yaml --coorddiag-manifest documents/experiments/2026-03-01_real_mapping_benchmark/ablation_optimization/manifests/atlasmtl_knn_geometry_eval_coorddiag.yaml --output-dir ~/tmp/atlasmtl_knn_geometry_eval_20260302 --devices cpu`

The output folder contains:

- `derived_inputs/` (including the derived no-obsm query copy)
- `generated_manifests/`
- `runs/`
- `metrics.json`
- `paper_tables/`
- `benchmark_report.md`

