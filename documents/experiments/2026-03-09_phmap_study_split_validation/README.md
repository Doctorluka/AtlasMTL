# PH-Map Study-Split Validation (`2026-03-09`)

Status: completed

This dossier records a PH-Map-only follow-up round that replaces the previous
`sample`-grouped split with a stricter `study`-grouped split.

Purpose:

- validate whether the PH-Map Phase 1 weighting result survives a
  study-isolated heldout protocol
- keep the experiment scoped to `PHMap_Lung_Full_v43_light`
- reuse the sixth-round AtlasMTL multi-level contract where possible

## Fixed contract

- dataset: `PHMap_Lung_Full_v43_light`
- label columns:
  - `anno_lv1`
  - `anno_lv2`
  - `anno_lv3`
  - `anno_lv4`
- `input_transform: binary`
- `optimizer_name: adamw`
- `weight_decay: 5e-5`
- `scheduler_name: null`
- `knn_correction: off`
- track: `gpu`

## Split design

- source reference: `/home/data/fhz/project/phmap_package/data/real_test/phmap/phmap_clean.h5ad`
- split key: `study`
- domain key: `study`
- prepared output root:
  - `/tmp/atlasmtl_benchmarks/2026-03-09/multilevel_annotation_study_split/PHMap_Lung_Full_v43_light/prepared/formal_split_v1/`

## Phase 1 comparison

Train configurations:

- `uniform_control = [1.0, 1.0, 1.0, 1.0]`
- `lv4strong_candidate = [0.2, 0.7, 1.5, 3.0]`

Predict settings:

- `enforce_hierarchy: true`
- `enforce_hierarchy: false`

Representative points:

- `build_100000_eval10k`
- `predict_100000_10000`

## Outcome

Phase 1 completed with `2` train runs and `8` predict evaluations.

Main result:

- `lv4strong_candidate` outperformed `uniform_control` in all four
  point-by-hierarchy conditions
- the weighting gain survived the stricter `study`-isolated heldout split
- `enforce_hierarchy=false` improved coverage and slightly improved or preserved
  finest-level metrics, at the cost of lower path consistency

Representative deltas:

- `predict_100000_10000 + hierarchy_on`
  - `anno_lv4 macro_f1`: `0.541423 -> 0.552978`
  - `full_path_accuracy`: `0.4570 -> 0.4639`
  - `coverage`: `0.7574 -> 0.7668`
- `build_100000_eval10k + hierarchy_on`
  - `anno_lv4 macro_f1`: `0.549254 -> 0.560272`
  - `full_path_accuracy`: `0.4548 -> 0.4591`
  - `coverage`: `0.7663 -> 0.7764`

Interpretation:

- the PH-Map fine-level weighting effect is reproducible under `study` split
- the hierarchy pass remains a post-hoc consistency tradeoff rather than the
  main source of PH-Map performance differences

## Repo-tracked outputs

- `results_summary/study_split_preparation_report.md`
- `results_summary/study_split_preparation_record.md`
- `results_summary/phase1_execution_record.md`
- `results_summary/phase1_levelwise.csv`
- `results_summary/phase1_hierarchy.csv`
- `results_summary/phase1_reliability.csv`
- `results_summary/phase1_comparison.csv`
- `results_summary/phase1_hierarchy_delta.csv`
- `results_summary/phase1_weight_and_hierarchy_ablation.md`
- `results_summary/phase2_execution_record.md`
- `results_summary/phase2_screen_summary.md`
- `results_summary/phase2_screen_comparison.csv`
- `results_summary/phase2_screen_best_config.json`
- `results_summary/phase2_seed_stability.md`
- `results_summary/phase2_seed_summary.csv`
- `results_summary/phase3_tradeoff_summary.md`
- `results_summary/phase3_tradeoff_levelwise.csv`
- `results_summary/phase3_tradeoff_hierarchy.csv`
- `results_summary/phase3_tradeoff_parent_child_breakdown.csv`
- `results_summary/phase3_tradeoff_subtree_breakdown.csv`
- `results_summary/phase3_tradeoff_by_study.csv`
- `results_summary/phase4_hotspot_refinement_summary.md`
- `results_summary/phase4_hotspot_refinement_comparison.csv`
- `results_summary/phase4_hotspot_refinement_parent_child_breakdown.csv`
- `results_summary/phase4_hotspot_refinement_subtree_breakdown.csv`
- `results_summary/phase4_hotspot_refinement_by_study.csv`
- `results_summary/phase4_execution_record.md`
- `results_summary/phase5_parent_conditioned_refinement_summary.md`
- `results_summary/phase5_parent_conditioned_refinement_comparison.csv`
- `results_summary/phase5_parent_conditioned_parent_child_breakdown.csv`
- `results_summary/phase5_parent_conditioned_subtree_breakdown.csv`
- `results_summary/phase5_parent_conditioned_by_study.csv`
- `results_summary/phase5_execution_record.md`
- `results_summary/phase6a_seed_comparison.csv`
- `results_summary/phase6a_seed_summary.csv`
- `results_summary/phase6a_seed_stability.md`
- `results_summary/phase6a_hotspot_sensitivity.csv`
- `results_summary/phase6a_hotspot_sensitivity.md`
- `results_summary/phase6a_by_study.csv`
- `results_summary/phase6a_by_study_summary.md`
- `results_summary/phase6a_parent_child_breakdown.csv`
- `results_summary/phase6a_execution_record.md`
- `results_summary/phase6b_comparison.csv`
- `results_summary/phase6b_seed_summary.csv`
- `results_summary/phase6b_parent_child_breakdown.csv`
- `results_summary/phase6b_by_study.csv`
- `results_summary/phase6b_summary.md`
- `results_summary/phase6b_execution_record.md`
- `results_summary/reranker_top6_v1/before_after_comparison.csv`
- `results_summary/reranker_top6_v1/before_after_parent_child_breakdown.csv`
- `results_summary/reranker_top6_v1/summary.md`
- `artifacts/reranker_top6_v1/hotspot_ranking.json`
- `artifacts/reranker_top6_v1/parent_conditioned_reranker_top6.pkl`
- `artifacts/reranker_top6_v1/parent_conditioned_reranker_top6.json`
- `artifacts/reranker_top6_v1/per_parent_reranker_summary.csv`
- `results_summary/phase6c_comparison.csv`
- `results_summary/phase6c_seed_summary.csv`
- `results_summary/phase6c_by_study.csv`
- `results_summary/phase6c_summary.md`
- `results_summary/phase6c_execution_record.md`
- `results_summary/phase7a_auto_reranker_pipeline/phase7a_auto_reranker_pipeline.csv`
- `results_summary/phase7a_auto_reranker_pipeline/phase7a_auto_reranker_parent_child_breakdown.csv`
- `results_summary/phase7a_auto_reranker_pipeline/phase7a_auto_reranker_pipeline.md`
- `results_summary/phase7b_hotspot_selection_comparison/phase7b_hotspot_selection_comparison.csv`
- `results_summary/phase7b_hotspot_selection_comparison/phase7b_selection_manifest.csv`
- `results_summary/phase7b_hotspot_selection_comparison/phase7b_hotspot_selection_comparison.md`
- `results_summary/phase7c_top6_vs_top8/phase7c_top6_vs_top8_seed_stability.csv`
- `results_summary/phase7c_top6_vs_top8/phase7c_top6_vs_top8_seed_summary.csv`
- `results_summary/phase7c_top6_vs_top8/phase7c_top6_vs_top8_by_study.csv`
- `results_summary/phase7c_top6_vs_top8/phase7c_top6_vs_top8_parent_overlap.csv`
- `results_summary/phase7c_top6_vs_top8/phase7c_top6_vs_top8_summary.md`
- `results_summary/phase7c_top6_vs_top8/phase7c_execution_record.md`
- `artifacts/phase7a_auto_reranker_pipeline/hotspot_ranking.json`
- `artifacts/phase7a_auto_reranker_pipeline/refinement_plan.json`
- `artifacts/phase7a_auto_reranker_pipeline/parent_conditioned_reranker_top6.pkl`
- `artifacts/phase7a_auto_reranker_pipeline/parent_conditioned_reranker_top6.json`
- `artifacts/phase7a_auto_reranker_pipeline/per_parent_reranker_summary.csv`
- `artifacts/phase7a_auto_reranker_pipeline/guardrail_decision.json`

## Scripts

- `scripts/generate_phase1_manifests.py`
- `scripts/run_phase1_gpu.sh`
- `scripts/collect_phase1_results.py`
- `scripts/generate_phase2_screen_manifests.py`
- `scripts/collect_phase2_screen_results.py`
- `scripts/generate_phase2_seed_manifests.py`
- `scripts/collect_phase2_seed_results.py`
- `scripts/run_phase2_gpu.sh`
- `scripts/generate_phase3_tradeoff_manifests.py`
- `scripts/run_phase3_tradeoff_gpu.sh`
- `scripts/collect_phase3_tradeoff_results.py`
- `scripts/run_phase4_hotspot_refinement.py`
- `scripts/run_phase4_hotspot_refinement.sh`
- `scripts/run_phase5_parent_conditioned_refinement.py`
- `scripts/run_phase5_parent_conditioned_refinement.sh`
- `scripts/run_phase6a_reranker_stability.py`
- `scripts/run_phase6a_reranker_stability.sh`
- `scripts/generate_phase6b_manifests.py`
- `scripts/run_phase6b_gpu.sh`
- `scripts/collect_phase6b_results.py`
- `scripts/materialize_reranker_top6_v1.py`
- `scripts/generate_phase6c_manifests.py`
- `scripts/run_phase6c_gpu.sh`
- `scripts/collect_phase6c_results.py`
- `scripts/run_phase7a_auto_reranker_pipeline.py`
- `scripts/run_phase7b_hotspot_selection_comparison.py`
- `scripts/run_phase7c_top6_vs_top8_seed_stability.py`

## Phase 2 outcome

Phase 2 tested finest-level imbalance handling on top of the `lv4strong`
weighting candidate.

Screened configs:

- `lv4strong_baseline`
- `lv4strong_plus_class_weight`
- `lv4strong_plus_class_balanced_sampling`
- `lv4strong_plus_both`

Single-seed screen result:

- `lv4strong_plus_class_weight` was selected as the best follow-up candidate
- on `predict_100000_10000`, `anno_lv4 macro_f1` improved from `0.565757` to
  `0.587630`
- `class_balanced_sampling` improved coverage strongly, but delivered a smaller
  finest-level gain
- the combined `class_weight + sampling` setting underperformed and was not
  advanced

Seed-stability result (`N=5`):

- `lv4strong_plus_class_weight` retained a higher finest-level mean and lower
  standard deviation than `lv4strong_baseline`
- on `predict_100000_10000`
  - `macro_f1 mean/std`: `0.56845 ± 0.01222` -> `0.58718 ± 0.00570`
  - `balanced_accuracy mean/std`: `0.56552 ± 0.01090` -> `0.58651 ± 0.00873`
- the main tradeoff is lower `full_path_accuracy`
  - `0.46904 ± 0.01141` -> `0.43836 ± 0.01042`

Interpretation:

- per-class weighting is the strongest current lever for finest-level PH-Map
  performance under `study` split
- it is also seed-stable on the main finest-level metrics
- however, it shifts the operating point away from best `full_path_accuracy`,
  so any default-promotion decision should state that tradeoff explicitly

## Phase 3 outcome

Phase 3 treated `lv4strong_plus_class_weight` as the new PH-Map baseline
candidate and reran `hierarchy on/off` against `lv4strong_baseline`, followed
by explicit parent-child and subtree error attribution.

Main result:

## Phase 7 outcome

Phase 7 shifted the PH-Map line from method exploration to operationalization.

Phase 7A result:

- the current best reranker path can be reproduced as a standard auto pipeline
- the auto pipeline now materializes:
  - hotspot ranking
  - refinement plan
  - saved reranker artifact
  - per-parent fit summary
  - guardrail decision
- for `predict_100000_10000 + hierarchy_on`, the auto pipeline passed the
  PH-Map guardrail:
  - `anno_lv4 macro_f1`: `0.582738 -> 0.585599`
  - `full_path_accuracy`: `0.4375 -> 0.4628`
  - `parent_correct_child_wrong_rate`: `0.1225 -> 0.0970`

Phase 7B result:

- the hotspot selection rule was compared on a small fixed grid:
  - `top4`
  - `top6`
  - `top8`
  - `cumulative_50pct`
  - `cumulative_60pct`
- in this seed-level PH-Map operational comparison, `top8` slightly outperformed
  `top6` on the main point:
  - `top6`: `macro_f1 = 0.585599`, `full_path_accuracy = 0.4628`
  - `top8`: `macro_f1 = 0.587033`, `full_path_accuracy = 0.4677`
- cumulative-contribution rules improved finest-level metrics but did not match
  the stronger full-path recovery of `top8`

Interpretation:

- the reranker path is now operationally standardized and no longer depends on
  hand-wired parent lists
- the best PH-Map hotspot rule may be slightly broader than the original `top6`
- `top8` advanced from candidate status to the primary default-rule contender
  that required a dedicated multi-seed confirmation round

## Phase 7C outcome

Phase 7C was a narrow stability confirmation round for the default hotspot rule.

Design:

- compare only `reranker_top6` vs `reranker_top8`
- reuse the current operational path:
  - `lv4strong + per-class weighting`
  - `+ auto parent-conditioned reranker`
- use the 5 already completed PH-Map seeds:
  - `101`
  - `17`
  - `2026`
  - `23`
  - `47`
- let each seed rediscover its own hotspot ranking from the seed-local baseline

Main result on `predict_100000_10000 + hierarchy_on`:

- `reranker_top6`
  - `macro_f1 = 0.587355 ± 0.002368`
  - `full_path_accuracy = 0.46664 ± 0.00493`
  - `parent_correct_child_wrong_rate = 0.09504 ± 0.00353`
- `reranker_top8`
  - `macro_f1 = 0.588557 ± 0.002093`
  - `full_path_accuracy = 0.47216 ± 0.00453`
  - `parent_correct_child_wrong_rate = 0.08926 ± 0.00310`

Additional stability observations:

- `top8` improved all three primary decision metrics over `top6`
- `top8` standard deviations did not worsen
- study-level gains stayed directionally consistent on the main heldout studies
- parent-set stability remained high:
  - `top6` had `6/6` overlap with seed-2026 in all 5 seeds
  - `top8` had `7/8` or `8/8` overlap with seed-2026 across seeds

Interpretation:

- the default-rule question is now settled more strongly than in Phase 7B
- `top8` is no longer just a single-seed candidate; it is the better PH-Map
  default hotspot rule under the current operational pipeline
- the current PH-Map operational default should therefore be updated from
  `top6` to `top8`

## Paper-facing close-out

The PH-Map line is now closed as the primary positive hard-case chapter result.

Final PH-Map operational path:

- `lv4strong + per-class weighting + auto parent-conditioned reranker_top8`

Paper-facing exports have been consolidated in:

- `results_summary/paper_ready/`

Interpretation:

- PH-Map now supports the full paper-facing chain:
  - hardest-case diagnosis
  - failed naive local fixes
  - successful parent-conditioned reranking
  - stable default-rule selection
- `6B` and `6C` remain useful research branches, but they do not compete with
  the reranker path for the operational default

- `lv4strong_plus_class_weight` remains better on finest-level metrics under
  both `hierarchy_on` and `hierarchy_off`
- the `full_path_accuracy` gap is still present
- the gap is explained mainly by more `anno_lv3 -> anno_lv4`
  child-level errors under otherwise correct parents, plus a moderate increase
  in off-mode path breaks

Representative attribution on `predict_100000_10000`:

- `hierarchy_on`
  - `anno_lv4 macro_f1`: `0.560001 -> 0.591375`
  - `full_path_accuracy`: `0.4562 -> 0.4443`
  - `parent_correct_child_wrong_rate` (`anno_lv4 <- anno_lv3`):
    `0.1003 -> 0.1175`
- `hierarchy_off`
  - `path_break_rate` (`anno_lv4 <- anno_lv3`):
    `0.0400 -> 0.0557`
  - `path_consistency_rate`:
    `0.9505 -> 0.9372`

Interpretation:

- the primary tradeoff category is `child discrimination tradeoff`
- the secondary tradeoff category is `hierarchy inconsistency tradeoff`
- `hierarchy_on` removes explicit path breaks, but does not remove the full-path
  gap because the main regression is not pure inconsistency; it is more often
  a wrong child under a correct parent
- subtree hotspots are concentrated in a limited set of `anno_lv3` parents,
  notably `CD8+ T`, `Mph alveolar`, `Pericyte`, and `CD4+ T`

## Phase 4 outcome

Phase 4 tested hotspot-local post-processing on top of
`lv4strong_plus_class_weight`:

- `hotspot_thresholding`
- `hotspot_temperature_scaling`
- `hotspot_thresholding_plus_temperature`

The hotspot parents were:

- `CD8+ T`
- `Mph alveolar`
- `Pericyte`
- `CD4+ T`

Result:

- no variant passed the Phase 4 guardrail
- all three refinements reduced `anno_lv4 macro_f1`
- none of them recovered `full_path_accuracy`
- none reduced `parent_correct_child_wrong_rate`

Representative result on `predict_100000_10000 + hierarchy_on`:

- baseline:
  - `macro_f1 = 0.591375`
  - `full_path_accuracy = 0.4443`
- `hotspot_temperature_scaling`:
  - `macro_f1 = 0.581127`
  - `full_path_accuracy = 0.4379`
- `hotspot_thresholding`:
  - `macro_f1 = 0.574235`
  - `full_path_accuracy = 0.4300`

Interpretation:

- the identified hotspot subtrees are real, but simple hotspot-local threshold
  tightening or temperature softening is not sufficient to fix the tradeoff
- the next justified step is no longer a generic post-processing tweak
- if further refinement is needed, it likely has to be a more structured
  parent-conditioned child model or a lightweight hierarchy-aware training
  modification

## Phase 5 outcome

Phase 5 moved from generic hotspot post-processing to a structured
parent-conditioned child refinement on top of `lv4strong_plus_class_weight`.

Method:

- fit one hotspot-specific logistic re-ranker per `anno_lv3` hotspot parent
- hotspot parents:
  - `CD8+ T`
  - `Mph alveolar`
  - `Pericyte`
  - `CD4+ T`
- training source for re-rankers: the same `study`-split reference train set
- inputs: restricted `anno_lv4` logits over legal children of the selected
  parent
- application rule: only rerank cells whose predicted `anno_lv3` falls into a
  hotspot parent

Result:

- the parent-conditioned re-ranker passed the Phase 5 acceptance criteria
- it improved both finest-level and full-path metrics
- it also reduced the main Phase 3 error mode:
  `parent_correct_child_wrong_rate`

Representative result on `predict_100000_10000 + hierarchy_on`:

- baseline:
  - `anno_lv4 macro_f1 = 0.591375`
  - `anno_lv4 balanced_accuracy = 0.595755`
  - `full_path_accuracy = 0.4443`
  - `parent_correct_child_wrong_rate = 0.1175`
- `parent_conditioned_reranker`:
  - `anno_lv4 macro_f1 = 0.602197`
  - `anno_lv4 balanced_accuracy = 0.622339`
  - `full_path_accuracy = 0.4630`
  - `parent_correct_child_wrong_rate = 0.0985`

Interpretation:

- this is the first follow-up that improves `full_path_accuracy` without giving
  back finest-level performance
- the gain supports the Phase 4 diagnosis that the residual PH-Map problem is a
  `parent-conditioned child decision` problem rather than a simple thresholding
  problem
- the most justified next step, if more improvement is needed, is now a
  train-time parent-conditioned child correction module rather than another
  generic post-processing sweep

## Phase 6A outcome

Phase 6A tested whether the inference-side parent-conditioned reranker is
stable across seeds, heldout studies, and hotspot coverage choices.

Design:

- reused the completed Phase 2 `lv4strong_plus_class_weight` seed bank
- seeds:
  - `101`
  - `17`
  - `2026`
  - `23`
  - `47`
- compared:
  - `baseline`
  - `reranker_top2`
  - `reranker_top4`
  - `reranker_top6`
- hotspot sets were ranked from Phase 3 baseline
  `predict_100000_10000 + hierarchy_on` by
  `parent_correct_child_wrong_rate * n_cells`

Result:

- all three reranker variants improved `full_path_accuracy` and reduced
  `parent_correct_child_wrong_rate`
- `top2` gave the cleanest finest-level gain, but smaller full-path recovery
- `top4` recovered substantially more full-path performance, with a small
  finest-level regression
- `top6` was the most stable overall winner across the `N=5` seed bank

Representative `predict_100000_10000 + hierarchy_on` seed means:

- baseline:
  - `anno_lv4 macro_f1 = 0.587177 ± 0.005695`
  - `full_path_accuracy = 0.43836 ± 0.01042`
  - `parent_correct_child_wrong_rate = 0.12348 ± 0.00986`
- `reranker_top6`:
  - `anno_lv4 macro_f1 = 0.587388 ± 0.002368`
  - `full_path_accuracy = 0.46666 ± 0.00493`
  - `parent_correct_child_wrong_rate = 0.09502 ± 0.00352`

Heldout-study result:

- gains were not driven by a single study
- on both `Slaven_Crnkovic_2022` and `Tijana_Tuhy_2025`, `reranker_top6`
  improved `macro_f1` and `full_path_accuracy` while reducing
  `parent_correct_child_wrong_rate`
- `Jonas_Schupp_2021` remained a degenerate heldout study with effectively zero
  finest-label support, so it does not drive the decision

Interpretation:

- the parent-conditioned reranker is not a single-seed artifact
- the mechanism is also not brittle to hotspot coverage, though wider coverage
  improved the tradeoff recovery more than the original manually selected top-4
  hotspot set
- this makes `parent-conditioned child refinement` stable enough to justify the
  next train-time upgrade

## Phase 6B outcome

Phase 6B tested whether the stable inference-side `reranker_top6` mechanism can
be internalized as a minimal train-time child correction module.

Compared variants:

- `baseline`
- `reranker_top6`
- `correction_joint`
- `correction_frozen_base`

Train-time correction settings:

- same PH-Map `study-split`
- same `lv4strong + per-class weighting` base contract
- hotspot set fixed to the Phase 6A `top6`
- two modes:
  - `joint`
  - `frozen_base`

Result:

- both train-time correction variants improved over the plain baseline
- neither train-time variant matched `reranker_top6`
- therefore no variant passed the Phase 6B reranker gate

Representative `predict_100000_10000 + hierarchy_on` seed means:

- baseline:
  - `anno_lv4 macro_f1 = 0.587177`
  - `full_path_accuracy = 0.43836`
  - `parent_correct_child_wrong_rate = 0.12348`
- `reranker_top6`:
  - `anno_lv4 macro_f1 = 0.587388`
  - `full_path_accuracy = 0.46666`
  - `parent_correct_child_wrong_rate = 0.09502`
- `correction_frozen_base`:
  - `anno_lv4 macro_f1 = 0.588199`
  - `full_path_accuracy = 0.45234`
  - `parent_correct_child_wrong_rate = 0.10946`
- `correction_joint`:
  - `anno_lv4 macro_f1 = 0.582636`
  - `full_path_accuracy = 0.44850`
  - `parent_correct_child_wrong_rate = 0.11080`

Interpretation:

- train-time internalization is directionally correct, because both
  correction-module variants still improve the main error mode relative to the
  plain baseline
- however, the current minimal train-time module underperforms the simpler
  inference-side `reranker_top6`
- `frozen_base` is the stronger of the two 6B variants and preserves
  finest-level performance better than `joint`
- the current evidence therefore supports keeping `reranker_top6` as the best
  operational module and not promoting the present train-time correction module
  as the new default

## reranker_top6 operational module v1

After Phase 6B, the operational priority was narrowed to solidifying the
current-best inference-side path rather than promoting the weaker train-time
internalization.

This v1 module was materialized on top of the Phase 2 `seed_2026`
`lv4strong_plus_class_weight` model and replayed through the public
`predict(..., refinement_config=...)` path.

Selection rule:

- source: `Phase 3 baseline`
- point: `predict_100000_10000 + hierarchy_on`
- score: `parent_correct_child_wrong_rate * n_cells`

Selected `top6` hotspot parents:

- `CD4+ T`
- `SMC de-differentiated`
- `Mph alveolar`
- `EC vascular`
- `Fibro adventitial`
- `CD8+ T`

Artifact outputs:

- `artifacts/reranker_top6_v1/hotspot_ranking.json`
- `artifacts/reranker_top6_v1/parent_conditioned_reranker_top6.pkl`
- `artifacts/reranker_top6_v1/parent_conditioned_reranker_top6.json`
- `artifacts/reranker_top6_v1/per_parent_reranker_summary.csv`
- `results_summary/reranker_top6_v1/before_after_comparison.csv`
- `results_summary/reranker_top6_v1/before_after_parent_child_breakdown.csv`

Representative replayed metrics:

- `build_100000_eval10k`
  - baseline:
    - `anno_lv4 macro_f1 = 0.572235`
    - `full_path_accuracy = 0.4325`
    - `parent_correct_child_wrong_rate = 0.1276`
  - `reranker_top6_v1`:
    - `anno_lv4 macro_f1 = 0.573133`
    - `full_path_accuracy = 0.4577`
    - `parent_correct_child_wrong_rate = 0.1024`
- `predict_100000_10000`
  - baseline:
    - `anno_lv4 macro_f1 = 0.582738`
    - `full_path_accuracy = 0.4375`
    - `parent_correct_child_wrong_rate = 0.1225`
  - `reranker_top6_v1`:
    - `anno_lv4 macro_f1 = 0.585599`
    - `full_path_accuracy = 0.4628`
    - `parent_correct_child_wrong_rate = 0.0970`

Interpretation:

- the saved reranker artifact reproduces the expected direction of improvement
  through the standard prediction entrypoint
- this closes the gap between the earlier dossier-local refinement scripts and
  a reusable, auditable operational module
- current PH-Map deployment should therefore continue to prefer
  `lv4strong + per-class weighting + reranker_top6 v1`

## Phase 6C outcome

Phase 6C tested a narrower train-time internalization path:

- kept only `frozen_base`
- changed correction inputs to a more reranker-like feature set
- added a local pairwise ranking term on hotspot parent subsets

Compared variants:

- `baseline`
- `reranker_top6`
- `correction_frozen_base_reranker_like`

Train-time settings:

- same PH-Map `study-split`
- same `lv4strong + per-class weighting` base contract
- hotspot set fixed to the Phase 6A `top6`
- correction feature mode: `reranker_like`
- added local rank loss with `rank_loss_weight = 0.2`

Result:

- 6C improved over the plain baseline on `full_path_accuracy` and
  `parent_correct_child_wrong_rate`
- 6C also improved over the 6B frozen-base version in overall closeness to the
  reranker
- but 6C still did not match `reranker_top6`, so the reranker gap remains open

Representative `predict_100000_10000 + hierarchy_on` seed means:

- baseline:
  - `anno_lv4 macro_f1 = 0.587177`
  - `balanced_accuracy = 0.586507`
  - `full_path_accuracy = 0.43836`
  - `parent_correct_child_wrong_rate = 0.12348`
- `correction_frozen_base_reranker_like`:
  - `anno_lv4 macro_f1 = 0.585776`
  - `balanced_accuracy = 0.583739`
  - `full_path_accuracy = 0.44884`
  - `parent_correct_child_wrong_rate = 0.11304`
- `reranker_top6`:
  - `anno_lv4 macro_f1 = 0.587388`
  - `balanced_accuracy = 0.617886`
  - `full_path_accuracy = 0.46666`
  - `parent_correct_child_wrong_rate = 0.09502`

Interpretation:

- making the frozen-base correction more reranker-like was directionally useful
- however, the remaining gap is still material:
  - `full_path_accuracy` is still `0.01782` below `reranker_top6`
  - `parent_correct_child_wrong_rate` is still `0.01802` above `reranker_top6`
  - `macro_f1` also remains slightly below the reranker path
- current evidence therefore still supports keeping `reranker_top6` as the
  best operational path, while treating train-time internalization as a
  methods-development branch rather than a promotion candidate
