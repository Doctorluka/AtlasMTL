# Experiment Dossiers

This directory stores repo-tracked benchmark and optimization dossiers.

Current rule for the paper-grade benchmark rollout:

- each reference dataset gets its own dossier
- each dossier owns its own manifests, scripts, notes, and summaries
- fairness is enforced within each scenario, not by forcing all references
  through one identical execution bundle

## Active dossiers

- `2026-03-01_real_mapping_benchmark`
  - early real-data benchmark and ablation baseline
- `2026-03-02_mtl_classifier_optimization`
  - AtlasMTL classifier optimization track
- `2026-03-03_phmap_benchmark`
  - pilot dossier for `PHMap_Lung_Full_v43_light`
- `2026-03-03_hlca_benchmark`
  - pilot dossier for `HLCA_Core`
- `2026-03-03_projectsvr_disco_benchmark`
  - pilot dossier for `DISCO_hPBMCs`
- `2026-03-04_projectsvr_mtca_benchmark`
  - second-wave scale-out dossier for `mTCA`
- `2026-03-04_projectsvr_cd4_benchmark`
  - second-wave scale-out dossier for `cd4`
- `2026-03-04_projectsvr_cd8_benchmark`
  - second-wave scale-out dossier for `cd8`
- `2026-03-04_projectsvr_vento_benchmark`
  - second-wave scale-out dossier for `Vento`
- `2026-03-05_formal_third_wave_hlca_benchmark`
  - formal third-wave pilot dossier for `HLCA_Core` (`train10k/test5k`, CPU/GPU split groups)
- `2026-03-06_scanvi_param_lock_benchmark`
  - pre-formal `scanvi` parameter confirmation dossier (GPU-only, cross-dataset stage-A/stage-B)
- `2026-03-07_atlasmtl_param_lock_benchmark`
  - pre-formal `atlasmtl` parameter confirmation dossier (CPU/GPU dual-track lock; Stage A/B completed on `2026-03-06`)
- `2026-03-06_formal_third_wave_scaling`
  - round-level formal scaling preparation dossier for third-wave main-panel and supplementary execution
- `2026-03-09_atlasmtl_low_cost_optimization`
  - fourth-round low-cost optimization dossier for atlasmtl default training refinement (`AdamW` / `weight_decay` / scheduler screening and confirmation)
- `2026-03-09_formal_atlasmtl_refresh`
  - fifth-round formal refresh dossier that reruns only AtlasMTL under the promoted default and compares against retained formal baseline rows
- `2026-03-09_phmap_study_split_validation`
  - PH-Map-only independent validation dossier that rebuilds the heldout split with `study` isolation and reruns the Phase 1 weighting comparison
- `2026-03-10_hlca_study_split_refinement_validation`
  - HLCA-specific validation dossier that rebuilds HLCA from raw `study` groups, confirms 5-level weighting, and tests first-pass AutoHotspot reranking
- `2026-03-10_parent_conditioned_reranker_paper_package`
  - paper-facing package that consolidates PH-Map and HLCA chapter tables and discussion notes for the parent-conditioned reranker line
- `2026-03-10_weight_activation_rule_validation`
  - validation dossier for the first error-driven policy that decides whether a dataset should leave `uniform` task weights

## Pilot rollout

The current first-wave pilots are:

- `PH-Map`
- `HLCA`
- `DISCO_hPBMCs`

Each pilot should first ship:

- one reference-heldout protocol
- one initial manifest draft
- one output-root convention
- one results-summary skeleton

External query validation can be added after the heldout protocol is reviewed.

Shared pre-execution checklist:

- `documents/protocols/pilot_benchmark_review_checklist.md`

Shared first-wave preprocessing contract:

- `documents/protocols/preprocessing_contract_first_wave.md`

## Second-wave scale-out

The current scale-out round expands from smoke validation to:

- all reference datasets
- larger heldout reference runs
- explicit preparation resource summaries
- nested `10k` / `5k` heldout outputs

Latest round-close status (`2026-03-05`):

- completed: `PHMap_Lung_Full_v43_light`, `DISCO_hPBMCs`, `mTCA`,
  `HLCA_Core`, `Vento`
- excluded in this round: `cd4`, `cd8` (raw-count contract blockers)

Execution plan:

- `plan/2026-03-04_second_wave_scaleout_benchmark_plan.md`
- `documents/experiments/2026-03-04_second_wave_round_status.md`
- `documents/experiments/2026-03-04_second_wave_execution_template.md`
- `documents/protocols/third_wave_fairness_protocol.md`

## Runtime caveats

When running comparator benchmarks in restricted/sandboxed environments, some
methods may silently degrade to serial execution even if `n_jobs > 1` is
configured. The most common signal is:

- `joblib will operate in serial mode` (`Errno 13: Permission denied`)

Implications:

- label-metric comparability is usually still valid
- runtime/throughput/core-usage comparisons are not strictly fair

For formal runtime fairness checks, rerun on an unrestricted server shell and
record effective thread/device usage in the run report.

## Formal third-wave scaling

The current formal benchmark round starts with a dedicated preparation phase:

- data audit
- group-aware split
- preprocessing and prepared subset materialization

Primary entry points:

- `plan/2026-03-06_formal_third_wave_scaling_plan.md`
- `documents/protocols/formal_third_wave_scaling_protocol.md`
- `documents/experiments/2026-03-06_formal_third_wave_execution_template.md`
- `documents/experiments/2026-03-06_formal_third_wave_round_status.md`
- `documents/experiments/2026-03-06_formal_third_wave_scaling/README.md`

## Recent round status

Latest dossier-level outcomes relevant to current atlasmtl defaults:

- `2026-03-07_atlasmtl_param_lock_benchmark`
  - completed
  - locked reproducible atlasmtl CPU/GPU benchmark grids before low-cost
    optimizer refinement
- `2026-03-09_atlasmtl_low_cost_optimization`
  - completed
  - Stage A selected `AdamW + wd=5e-5` as the only credible candidate
  - Stage B confirmed promotion using GPU-first evidence
  - `ReduceLROnPlateau` was rejected as a default

Current atlasmtl training default after the completed optimization round:

- `optimizer_name="adamw"`
- `weight_decay=5e-5`
- `scheduler_name=None`

Important interpretation caveat:

- CPU evidence from the optimization round was mixed and collected under
  `joblib_serial_fallback`
- non-degraded GPU Stage B confirmation is therefore the primary evidence for
  the default-promotion decision

- `2026-03-09_formal_atlasmtl_refresh`
  - completed
  - AtlasMTL-only formal refresh finished on `20` planned points
  - refresh did not clear the formal row-replacement gate
  - retained formal third-wave AtlasMTL baseline rows remain the paper-grade
    comparison rows
  - code default still remains `AdamW + weight_decay=5e-5`
  - practical interpretation: software default retained, manuscript-grade
    comparison rows unchanged
- `2026-03-09_phmap_study_split_validation`
  - completed
  - PH-Map current-best operational path is now:
    `lv4strong + per-class weighting + auto parent-conditioned reranker_top8`
  - `top8` passed the default-rule confirmation and replaced `top6`
  - train-time internalization remains a research branch, not the default path
- `2026-03-10_hlca_study_split_refinement_validation`
  - in progress
  - HLCA `study`-split and 5-level weighting confirmation are complete
  - current best HLCA base weighting is `uniform`, not PH-Map-style finest-level upweighting
  - first-pass auto reranker validation is mixed and has not promoted reranking to an HLCA operational default
  - narrowed `top4` vs `top6` rule comparison remains mixed; HLCA is currently retained as a stress-test case rather than a second positive reranker dataset
- `2026-03-10_parent_conditioned_reranker_paper_package`
  - completed
  - consolidates PH-Map and HLCA chapter tables into a single paper-facing package
  - PH-Map is treated as the finalized positive case
  - HLCA is currently carried as a mixed first-pass secondary validation case
