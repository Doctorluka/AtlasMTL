# HLCA Study-Split Refinement Validation (`2026-03-10`)

Status: in_progress

This dossier is the publication-oriented follow-up round that extends the
PH-Map AutoHotspot reranker line to `HLCA_Core`.

Purpose:

- rebuild HLCA prepared inputs from the raw reference using a stricter
  `study`-grouped split
- confirm a dataset-specific 5-level task-weight schedule for HLCA before any
  reranker validation
- prepare paper-ready outputs for a second deep-hierarchy dataset that can be
  shown alongside PH-Map

## Fixed framing

- dataset: `HLCA_Core`
- source reference:
  - `/home/data/fhz/project/phmap_package/data/real_test/HLCA/hlca_clean.h5ad`
- split key: `study`
- domain key: `study`
- label columns:
  - `ann_level_1`
  - `ann_level_2`
  - `ann_level_3`
  - `ann_level_4`
  - `ann_level_5`
- finest level: `ann_level_5`

## Planned stages

1. `study`-split preparation from raw HLCA
2. 5-level weighting confirmation:
   - `uniform`
   - `mild_lv5`
   - `strong_lv5`
3. best-base reranker mechanism validation
4. narrowed reranker-rule decision round
   - `top4`
   - `top6`
5. paper-ready panel and supplementary table export

## Expected prepared root

- `/tmp/atlasmtl_benchmarks/2026-03-10/hlca_study_split_refinement/HLCA_Core/prepared/formal_split_v1/`

## Scripts

- `scripts/prepare_hlca_study_split.py`
- `scripts/run_prepare_hlca_study_split.sh`
- `scripts/generate_hlca_weight_confirmation_manifests.py`
- `scripts/run_hlca_weight_confirmation_gpu.sh`
- `scripts/collect_hlca_weight_confirmation_results.py`

## Key intended outputs

- `results_summary/study_split_preparation_report.md`
- `results_summary/study_split_preparation_record.md`
- `results_summary/hlca_weight_confirmation.csv`
- `results_summary/hlca_weight_confirmation.md`
- `results_summary/hlca_auto_reranker_validation/hlca_main_comparison.csv`
- `results_summary/hlca_auto_reranker_validation/hlca_error_decomposition.csv`
- `results_summary/hlca_auto_reranker_validation/hlca_auto_reranker_validation.md`
- `results_summary/hlca_reranker_rule_comparison/hlca_reranker_rule_comparison.csv`
- `results_summary/hlca_reranker_rule_comparison/hlca_reranker_rule_summary.md`

## Current status

Completed:

1. `study`-split preparation from raw HLCA
2. 5-level weighting confirmation
   - winner: `uniform`
   - HLCA does not inherit the PH-Map finest-level upweight schedule
3. first-pass auto reranker mechanism validation
   - best base config: `uniform`
   - target edge: `ann_level_4 -> ann_level_5`
   - selected hotspot parents:
     - `Alveolar macrophages`
     - `Interstitial macrophages`
     - `Goblet`
     - `Club`
     - `Multiciliated`
     - `AT2`
4. narrowed reranker-rule decision round
   - compared `reranker_top4` vs `reranker_top6`
   - both remain mixed-evidence variants

Current interpretation:

- HLCA first-pass reranking is directionally mixed rather than decisively positive
- on `predict_100000_10000 + hierarchy_on`, auto reranking improves `ann_level_5 macro_f1`
  from `0.688732` to `0.693015`
- but `full_path_accuracy` drops from `0.8239` to `0.8200`
- and `parent_correct_child_wrong_rate` rises from `0.0334` to `0.0371`
- the PH-Map-style guardrail therefore fails on HLCA in this first validation
- the narrowed `top4` vs `top6` rule comparison does not reverse this result:
  - `top4` improves `ann_level_5 macro_f1` to `0.692970` but lowers `full_path_accuracy` to `0.8199`
  - `top6` improves `ann_level_5 macro_f1` to `0.693015` but lowers `full_path_accuracy` to `0.8200`
  - both variants also raise `parent_correct_child_wrong_rate`

Operational implication:

- HLCA currently supports the weighting conclusion (`uniform` remains best)
- but does not yet support promotion of auto parent-conditioned reranking to an HLCA operational default
- HLCA is currently better framed as a mixed-evidence stress-test dataset than as a second positive reranker generalization case
