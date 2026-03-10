# 2026-03-09 AtlasMTL Multi-Level Annotation Benchmark

Status: completed

Sub-tracks:

- `v1`
  - completed uniform-weight multi-level benchmark
- `v2_weighted_gpu`
  - weighted GPU-only redesign track that preserves v1 outputs

## Purpose

This dossier defines the sixth-round AtlasMTL-only multi-level annotation
benchmark. The goal is to evaluate AtlasMTL as a multi-level `sc -> sc`
reference mapping framework rather than re-running a shared single-level
comparator benchmark.

## Dataset roster

- `HLCA_Core`
  - levels: `ann_level_1 -> ann_level_5`
- `PHMap_Lung_Full_v43_light`
  - levels: `anno_lv1 -> anno_lv4`
- `DISCO_hPBMCs`
  - levels: `cell_type -> cell_subtype`
- `mTCA`
  - levels: `Cell_type_level1 -> Cell_type_level3`

## Fixed experimental contract

- AtlasMTL only
- tracks: `cpu_core`, `gpu`
- representative points:
  - `build_100000_eval10k`
  - `predict_100000_10000`
- `knn_correction: off`
- `optimizer_name: adamw`
- `weight_decay: 5e-5`
- `scheduler_name: null`
- `input_transform: binary`

## Outputs

- `results_summary/dataset_hierarchy_inventory.csv`
- `results_summary/levelwise_performance.csv`
- `results_summary/hierarchy_performance.csv`
- `results_summary/reliability_performance.csv`
- `results_summary/multilevel_summary.md`
- `results_summary/multilevel_decision_note.md`
- `results_summary/multilevel_experiment_report.md`
- `results_summary/hierarchy_aware_discussion_note.md`
- `results_summary/v2_weighted_gpu/levelwise_performance.csv`
- `results_summary/v2_weighted_gpu/hierarchy_performance.csv`
- `results_summary/v2_weighted_gpu/reliability_performance.csv`
- `results_summary/v2_weighted_gpu/comparison_vs_v1_uniform_gpu.csv`
- `results_summary/v2_weighted_gpu/v2_weighted_gpu_report.md`
- `results_summary/v2_weighted_gpu/v2_weighted_gpu_decision_note.md`
- `results_summary/phase1_phmap_weight_hierarchy/phase1_levelwise.csv`
- `results_summary/phase1_phmap_weight_hierarchy/phase1_hierarchy.csv`
- `results_summary/phase1_phmap_weight_hierarchy/phase1_reliability.csv`
- `results_summary/phase1_phmap_weight_hierarchy/phase1_comparison.csv`
- `results_summary/phase1_phmap_weight_hierarchy/phase1_hierarchy_delta.csv`
- `results_summary/phase1_phmap_weight_hierarchy/phase1_weight_and_hierarchy_ablation.md`

## Scripts

- `scripts/generate_dataset_hierarchy_inventory.py`
- `scripts/generate_hierarchy_rules.py`
- `scripts/generate_multilevel_manifests.py`
- `scripts/run_multilevel_cpu.sh`
- `scripts/run_multilevel_gpu.sh`
- `scripts/collect_multilevel_results.py`
- `scripts/generate_multilevel_v2_weighted_gpu_manifests.py`
- `scripts/run_multilevel_v2_weighted_gpu.sh`
- `scripts/collect_multilevel_v2_weighted_gpu_results.py`
- `scripts/generate_phase1_phmap_weight_hierarchy_manifests.py`
- `scripts/run_phase1_phmap_weight_hierarchy_gpu.sh`
- `scripts/collect_phase1_phmap_weight_hierarchy_results.py`

## Runtime root

Recommended temporary runtime root:

`/tmp/atlasmtl_benchmarks/2026-03-09/multilevel_annotation/`

## Interpretation rules

- Main analysis is `multi-level first`.
- Primary tables should emphasize:
  - level-wise label quality
  - hierarchy-aware path quality
  - finest-level reliability and abstention behavior
- This round does not modify the retained third-wave manuscript comparison rows.
- Actual execution status:
  - `16/16` runs completed successfully
  - CPU runs are marked degraded for runtime fairness because of
    `joblib` serial fallback in the restricted environment
  - GPU runs completed cleanly and should be treated as the primary
    runtime-quality evidence for this round
- v2 redesign rule:
  - the weighted GPU re-run is tracked separately under `v2_weighted_gpu`
  - v1 files remain frozen and should continue to be cited as the original
    sixth-round execution record
