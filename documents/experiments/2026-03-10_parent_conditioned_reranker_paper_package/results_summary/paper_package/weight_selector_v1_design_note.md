# Weight Selector V1 Design Note

This note describes the next step after the current activation-rule validation.

## Positioning

- The current framework can now decide whether a dataset should leave `uniform` task weights.
- The next step is not a free weight searcher.
- It is a small candidate selector that is only activated when `activate_nonuniform_weighting = true`.

## Candidate selector v1

- 4-level datasets:
  - `uniform = [1, 1, 1, 1]`
  - `mild_lv4 = [0.7, 0.8, 1.0, 1.8]`
  - `strong_lv4 = [0.2, 0.7, 1.5, 3.0]`
- 5-level datasets:
  - `uniform = [1, 1, 1, 1, 1]`
  - `mild_lv5 = [0.7, 0.8, 1.0, 1.2, 2.0]`
  - `strong_lv5 = [0.4, 0.6, 0.9, 1.3, 3.0]`

## Decision rule

- Only run the selector on datasets that passed the activation rule.
- Choose the winner using a multi-objective guardrail:
  - maximize finest-level `macro_f1`
  - prefer higher `full_path_accuracy`
  - prefer lower `parent_correct_child_wrong_rate`
- Reject any candidate that raises finest-level metrics but clearly worsens hierarchy-structured error.

## Intended paper role

- activation policy belongs in Methods / Supplementary Methods as a framework policy
- selector v1 is a narrow operational policy, not a global optimizer
