# 2026-03-01 Real Mapping Benchmark Setup

This change adds a dedicated real-data benchmark dossier for the sampled
reference/query test adatas.

Added repo-tracked experiment structure:

- `documents/experiments/2026-03-01_real_mapping_benchmark/`
  - `plan/`
  - `scripts/`
  - `manifests/`
  - `results_summary/`
  - `notes/`

Added scripts for this run:

- preprocessing and audit generation
- hierarchy-manifest generation
- single-level benchmark execution
- multi-level AtlasMTL execution
- local CellTypist model training helper

Added run records and summaries for:

- raw-data audit
- single-level benchmark results
- multi-level AtlasMTL results
- overall assessment and remaining blockers

Runtime storage policy for this run:

- repo keeps the reproducible dossier
- large run artifacts are written to
  `~/tmp/atlasmtl_real_mapping_benchmark_20260301/`
