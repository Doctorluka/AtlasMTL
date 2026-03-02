# Ablation Results Summary

This directory stores repo-tracked summaries for the completed AtlasMTL
ablation round.

Runtime artifacts remain in the private workspace:

- `~/tmp/atlasmtl_real_mapping_benchmark_20260301/ablation_optimization/metrics.json`
- `~/tmp/atlasmtl_real_mapping_benchmark_20260301/ablation_optimization/benchmark_report.md`
- `~/tmp/atlasmtl_real_mapping_benchmark_20260301/ablation_optimization/paper_tables/`

Primary summary files:

- `atlasmtl_ablation_summary.md`
- `atlasmtl_ablation_discussion.md`
- `interim_hvg_weight_comparison.md`

Planned follow-up summaries:

- `hvg_tradeoff_interim.md`
- `hvg_tradeoff_recommendation.md`
- `weight_scan_interim.md`
- `weight_scan_recommendation.md`

The interim comparison note is for internal decision-making only. It summarizes
the current `whole` / `hvg3000` / `hvg6000` and `uniform` / `phmap` contrast,
but it is not the final paper-facing benchmark conclusion.

The new HVG and weight-search summaries are also internal benchmark materials.
They record completed optimization runs and current operational
recommendations, but they still require cross-dataset confirmation before any
paper-facing conclusion is frozen.
