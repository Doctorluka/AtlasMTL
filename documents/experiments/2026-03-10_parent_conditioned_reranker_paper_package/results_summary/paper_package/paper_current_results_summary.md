# Current Results Summary For Expert Discussion

## PH-Map

- Final operational path is now fixed as `lv4strong + per-class weighting + auto parent-conditioned reranker_top8`.
- PH-Map provides the finalized positive hard-case result.
- On `predict_100000_10000 + hierarchy_on`, the paper-ready mean results are:
  - base + class weighting: macro_f1 `0.587177 ± 0.005695`, full_path `0.43836 ± 0.01042`, parent_correct_child_wrong `0.12348 ± 0.00986`
  - + auto reranker_top8: macro_f1 `0.588557 ± 0.002093`, full_path `0.47216 ± 0.00453`, parent_correct_child_wrong `0.08926 ± 0.00310`
- `top8` has passed the default-rule confirmation and replaces `top6` as the PH-Map default hotspot rule.

## HLCA

- HLCA `study`-split preprocessing and weighting confirmation are complete.
- HLCA does not inherit the PH-Map finest-level upweighting schedule; `uniform` is currently the best base configuration.
- First-pass auto reranker validation on `ann_level_4 -> ann_level_5` yields mixed evidence.
- A narrowed `top4` vs `top6` reranker-rule comparison does not change that conclusion.
- On `predict_100000_10000 + hierarchy_on`:
  - baseline uniform: macro_f1 `0.688732`, full_path `0.8239`, parent_correct_child_wrong `0.0334`
  - + auto reranker_top4: macro_f1 `0.692970`, full_path `0.8199`, parent_correct_child_wrong `0.0374`
  - + auto reranker_top6: macro_f1 `0.693015`, full_path `0.8200`, parent_correct_child_wrong `0.0371`
- HLCA therefore improves finest-level macro-F1 but fails the PH-Map-style guardrail because full-path declines and the main error mode worsens.

## Weight activation policy

- A first-pass error-driven activation rule has now been validated on three contrasting multi-level cases.
- `PH-Map` is correctly classified as `activate_nonuniform_weighting = true`.
- `HLCA` is correctly classified as `activate_nonuniform_weighting = false`.
- `mTCA` also stays on `uniform`, providing a shallower third-point sanity check rather than another hard-case trigger.
- This supports the framework policy claim that non-uniform task weighting should be treated as a dataset-adaptive option rather than as a universal default schedule.

## Current paper interpretation

- PH-Map is a strong positive case for the chapter claim.
- HLCA currently supports the dataset-specific weighting claim, but remains a mixed-evidence stress test for reranker transfer rather than a second positive reranker case.
- The weighting story is now stronger than before because the chapter can distinguish two separate claims: the schedule itself is dataset-specific, and the choice to leave uniform can be made by an error-driven activation policy.
- At the moment, the chapter can robustly claim a positive operational module on PH-Map and a nontrivial second-dataset stress test on HLCA.
