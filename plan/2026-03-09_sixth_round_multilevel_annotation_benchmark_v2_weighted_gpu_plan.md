# Sixth-Round V2 Plan: Weighted GPU Multi-Level Re-Run

## Objective

Add a weighted GPU-only redesign track under the existing sixth-round
multi-level annotation benchmark without overwriting the original v1 uniform
results.

## Scope

- Datasets:
  - `HLCA_Core`
  - `PHMap_Lung_Full_v43_light`
  - `DISCO_hPBMCs`
  - `mTCA`
- Track:
  - `gpu`
- Representative points:
  - `build_100000_eval10k`
  - `predict_100000_10000`
- Total runs:
  - `8`

## Fixed configuration

- `train.input_transform: binary`
- `train.optimizer_name: adamw`
- `train.weight_decay: 5e-5`
- `train.scheduler_name: null`
- `train.task_weights: [0.2, 0.7, 1.5, 3.0]`
- `train.num_epochs: 50`
- `train.learning_rate: 3e-4`
- `predict.knn_correction: off`
- `predict.enforce_hierarchy: true`

## Deliverables

- `manifests/multilevel_v2_weighted_gpu/manifest_index.json`
- `scripts/generate_multilevel_v2_weighted_gpu_manifests.py`
- `scripts/run_multilevel_v2_weighted_gpu.sh`
- `scripts/collect_multilevel_v2_weighted_gpu_results.py`
- `results_summary/v2_weighted_gpu/levelwise_performance.csv`
- `results_summary/v2_weighted_gpu/hierarchy_performance.csv`
- `results_summary/v2_weighted_gpu/reliability_performance.csv`
- `results_summary/v2_weighted_gpu/comparison_vs_v1_uniform_gpu.csv`
- `results_summary/v2_weighted_gpu/comparison_vs_v1_uniform_gpu.md`
- `results_summary/v2_weighted_gpu/v2_weighted_gpu_report.md`
- `results_summary/v2_weighted_gpu/v2_weighted_gpu_decision_note.md`

## Notes

- v1 results remain frozen and are not edited or replaced.
- v2 is judged primarily against v1 GPU rows.
- CPU backfill is deferred unless v2 GPU shows a strong enough signal.
