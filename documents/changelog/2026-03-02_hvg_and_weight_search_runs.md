# 2026-03-02 HVG and Weight Search Runs

This changelog records the completion of the first dedicated HVG tradeoff run
and task-weight search run under the AtlasMTL ablation optimization dossier.

## Added

- dedicated manifests for:
  - `hvg_tradeoff`
  - `weight_scan`
- dedicated runner scripts for:
  - `run_atlasmtl_hvg_tradeoff.py`
  - `run_atlasmtl_weight_scan.py`
- dedicated plotting scripts for:
  - `plot_hvg_tradeoff.py`
  - `plot_weight_scan.py`
- shared ablation helper:
  - `ablation_common.py`
- repo-tracked interim/recommendation summaries for both search directions

## Completed runtime outputs

Private runtime bundles were written to:

- `~/tmp/atlasmtl_real_mapping_benchmark_20260301/ablation_optimization/hvg_tradeoff/`
- `~/tmp/atlasmtl_real_mapping_benchmark_20260301/ablation_optimization/weight_scan/`

Both bundles now contain:

- `metrics.json`
- `paper_tables/`
- `benchmark_report.md`
- `analysis/`

## Current internal recommendations

- HVG tradeoff:
  - `cpu`: `hvg5000`
  - `cuda`: `hvg6000`
- weight search:
  - `cpu`: `ratio_1.6`
  - `cuda`: `lv4strong_a = [0.2, 0.7, 1.5, 3.0]`

## Notes

- these are benchmark-internal recommendations only
- they are intended to guide the next benchmark round
- they are not yet final paper-facing conclusions
