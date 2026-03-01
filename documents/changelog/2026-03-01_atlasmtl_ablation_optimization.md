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
