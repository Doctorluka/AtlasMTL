# 2026-03-01 AtlasMTL ablation optimization support

## Summary

Added the next-round AtlasMTL ablation support needed to compare:

- `task_weights`: uniform vs PH-Map-style
- `feature_space`: whole vs HVG 3000 / 6000
- `input_transform`: binary vs raw-count float
- `device`: CPU, plus CUDA only after an execution gate succeeds

## Code changes

- atlasmtl benchmark runner now supports AtlasMTL-specific counts-layer inputs
  without changing the core preprocessing contract
- AtlasMTL benchmark payloads now record:
  - `variant_name`
  - `ablation_config`
  - explicit matrix sources for reference/query
  - task weights used during training
- paper-table exports now emit:
  - `atlasmtl_ablation_accuracy`
  - `atlasmtl_ablation_resources`
  - `atlasmtl_ablation_tradeoff`

## Experiment dossier changes

Added:

- `documents/experiments/2026-03-01_real_mapping_benchmark/ablation_optimization/`
  - base manifest
  - CUDA gate script
  - AtlasMTL ablation runner
  - execution notes and locked plan

## Protocol clarification

- `binary` vs `float` AtlasMTL ablations are now defined on the same
  `layers["counts"]` source
- `float` means raw counts cast to `float32`, not log-normalized or log1p
- GPU benchmark variants require a benchmark-entry CUDA gate

## Execution completion

The full `24`-run AtlasMTL ablation grid has now been executed on the sampled
real benchmark bundle:

- devices: `cpu`, `cuda`
- feature spaces: `whole`, `hvg3000`, `hvg6000`
- input transforms: `binary`, `float`
- task-weight schemes: `uniform`, `phmap`

Top result on `anno_lv4`:

- `atlasmtl_cuda_hvg6000_binary_phmap`
  - accuracy `0.7730`
  - macro-F1 `0.6720`

High-level findings:

- `binary` outperformed `float`
- `phmap` weights outperformed `uniform`
- `hvg6000` produced the best observed accuracy/resource tradeoff
- CPU and GPU delivered near-identical average accuracy, while GPU reduced
  training time substantially

Completed runtime outputs were written to the private workspace:

- `~/tmp/atlasmtl_real_mapping_benchmark_20260301/ablation_optimization/metrics.json`
- `~/tmp/atlasmtl_real_mapping_benchmark_20260301/ablation_optimization/benchmark_report.md`
- `~/tmp/atlasmtl_real_mapping_benchmark_20260301/ablation_optimization/paper_tables/`
