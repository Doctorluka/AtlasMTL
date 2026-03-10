# Sixth-Round Plan: AtlasMTL Multi-Level Annotation Benchmark

## Objective

Establish manuscript-grade method evidence for AtlasMTL as a multi-level `sc -> sc`
reference mapping framework. This round evaluates AtlasMTL-only multi-level
annotation behavior rather than re-running the shared single-level comparator
benchmark.

## Scope

- Datasets:
  - `HLCA_Core`
  - `PHMap_Lung_Full_v43_light`
  - `DISCO_hPBMCs`
  - `mTCA`
- Tracks:
  - `cpu_core`
  - `gpu`
- Representative points:
  - `build_100000_eval10k`
  - `predict_100000_10000`
- Methods:
  - `atlasmtl` only

## Fixed configuration

- `train.input_transform: binary`
- `train.optimizer_name: adamw`
- `train.weight_decay: 5e-5`
- `train.scheduler_name: null`
- `predict.knn_correction: off`
- `predict.enforce_hierarchy: true`

## Deliverables

- Dossier:
  - `documents/experiments/2026-03-09_multilevel_annotation_benchmark/README.md`
- Inventory:
  - `results_summary/dataset_hierarchy_inventory.csv`
- Manifests:
  - `manifests/multilevel/manifest_index.json`
  - per-run YAML manifests
  - per-dataset hierarchy JSON files
- Scripts:
  - `generate_dataset_hierarchy_inventory.py`
  - `generate_hierarchy_rules.py`
  - `generate_multilevel_manifests.py`
  - `run_multilevel_cpu.sh`
  - `run_multilevel_gpu.sh`
  - `collect_multilevel_results.py`
- Results:
  - `results_summary/levelwise_performance.csv`
  - `results_summary/hierarchy_performance.csv`
  - `results_summary/reliability_performance.csv`
  - `results_summary/multilevel_summary.md`
  - `results_summary/multilevel_decision_note.md`

## Decision focus

This round should answer:

1. Does AtlasMTL retain useful label quality from coarse to fine levels?
2. Does full-path performance remain acceptable on deep hierarchies?
3. Do abstention and confidence-derived reliability metrics support the
   framework-level claim of reliable multi-level annotation?

## Notes

- KNN is explicitly excluded from this round.
- This round does not replace retained third-wave manuscript comparison rows.
- CPU runtime evidence may still be degraded under restricted execution and must
  be labeled if `joblib` falls back to serial mode.
