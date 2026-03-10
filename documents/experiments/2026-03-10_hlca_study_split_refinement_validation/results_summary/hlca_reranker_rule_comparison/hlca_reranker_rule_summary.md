# HLCA Reranker Rule Comparison

- best base config: `uniform`
- compared rules: `top4` vs `top6` on `ann_level_4 -> ann_level_5`
- minimum parent cells: `200`

## Target point

- baseline: macro_f1 `0.688732`, full_path `0.8239`, parent_correct_child_wrong `0.0334`
- reranker_top4: macro_f1 `0.692970`, full_path `0.8199`, parent_correct_child_wrong `0.0374`
- reranker_top6: macro_f1 `0.693015`, full_path `0.8200`, parent_correct_child_wrong `0.0371`

## Verdict

- HLCA retained as mixed-evidence stress test

## Outputs

- `hlca_reranker_rule_comparison.csv`
- `hlca_reranker_rule_by_study.csv`
- `hlca_reranker_rule_guardrails.csv`
