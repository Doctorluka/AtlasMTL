# 2026-03-01 Real Mapping Benchmark

This dossier stores the repo-tracked materials for the real-data benchmark run
using:

- reference: `data/test_adatas/sampled_adata_10k.h5ad`
- query: `data/test_adatas/sampled_adata_3000.h5ad`

Scope:

- single-level formal comparator benchmark on `anno_lv4`
- multi-level AtlasMTL run on `anno_lv1` to `anno_lv4`
- hierarchy-aware evaluation
- explicit exclusion of KNN correction for this run because the datasets do
  not provide reference/query coordinate targets in `obsm`

Storage split:

- repo-tracked specifications, scripts, manifests, and summaries live here
- large runtime artifacts live under
  `~/tmp/atlasmtl_real_mapping_benchmark_20260301/`

Directory layout:

- `plan/`
  - locked execution plan used for this run
- `scripts/`
  - reproducible preparation and run scripts
- `manifests/`
  - dataset manifests and hierarchy config used by the run
- `results_summary/`
  - compact human-readable summaries and evaluation notes
- `notes/`
  - environment notes, exclusions, and issues
