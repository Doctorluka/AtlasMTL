# PH-Map Study-Split Phase 2 Screen

- evaluated configs: `4`
- evaluation rows: `8`

## Finest-Level Rows

| config_name                            | point                |   macro_f1 |   balanced_accuracy |   accuracy |   coverage |   unknown_rate |   covered_accuracy |
|:---------------------------------------|:---------------------|-----------:|--------------------:|-----------:|-----------:|---------------:|-------------------:|
| lv4strong_baseline                     | build_100000_eval10k |   0.573018 |            0.572572 |     0.4814 |     0.7973 |         0.2027 |           0.603788 |
| lv4strong_baseline                     | predict_100000_10000 |   0.565757 |            0.560818 |     0.48   |     0.792  |         0.208  |           0.606061 |
| lv4strong_plus_both                    | build_100000_eval10k |   0.501565 |            0.537612 |     0.3817 |     0.8093 |         0.1907 |           0.471642 |
| lv4strong_plus_both                    | predict_100000_10000 |   0.521184 |            0.534554 |     0.3912 |     0.8091 |         0.1909 |           0.4835   |
| lv4strong_plus_class_balanced_sampling | build_100000_eval10k |   0.557923 |            0.596795 |     0.4506 |     0.8804 |         0.1196 |           0.511813 |
| lv4strong_plus_class_balanced_sampling | predict_100000_10000 |   0.575706 |            0.602086 |     0.4576 |     0.8758 |         0.1242 |           0.522494 |
| lv4strong_plus_class_weight            | build_100000_eval10k |   0.566095 |            0.590616 |     0.4453 |     0.7767 |         0.2233 |           0.573323 |
| lv4strong_plus_class_weight            | predict_100000_10000 |   0.58763  |            0.593141 |     0.4538 |     0.7714 |         0.2286 |           0.588281 |

## Baseline Comparison

| config_name                            | point                |   macro_f1 |   delta_macro_f1_vs_baseline |   full_path_accuracy |   delta_full_path_accuracy_vs_baseline |   coverage |   delta_coverage_vs_baseline |
|:---------------------------------------|:---------------------|-----------:|-----------------------------:|---------------------:|---------------------------------------:|-----------:|-----------------------------:|
| lv4strong_baseline                     | build_100000_eval10k |   0.573018 |                   0          |               0.4717 |                                 0      |     0.7973 |                       0      |
| lv4strong_plus_both                    | build_100000_eval10k |   0.501565 |                  -0.0714537  |               0.3759 |                                -0.0958 |     0.8093 |                       0.012  |
| lv4strong_plus_class_balanced_sampling | build_100000_eval10k |   0.557923 |                  -0.0150957  |               0.4403 |                                -0.0314 |     0.8804 |                       0.0831 |
| lv4strong_plus_class_weight            | build_100000_eval10k |   0.566095 |                  -0.00692351 |               0.4385 |                                -0.0332 |     0.7767 |                      -0.0206 |
| lv4strong_baseline                     | predict_100000_10000 |   0.565757 |                   0          |               0.472  |                                 0      |     0.792  |                       0      |
| lv4strong_plus_both                    | predict_100000_10000 |   0.521184 |                  -0.0445728  |               0.3844 |                                -0.0876 |     0.8091 |                       0.0171 |
| lv4strong_plus_class_balanced_sampling | predict_100000_10000 |   0.575706 |                   0.00994961 |               0.4476 |                                -0.0244 |     0.8758 |                       0.0838 |
| lv4strong_plus_class_weight            | predict_100000_10000 |   0.58763  |                   0.0218734  |               0.4437 |                                -0.0283 |     0.7714 |                      -0.0206 |

## Selected Candidate

{
  "best_config_name": "lv4strong_plus_class_weight",
  "coverage": 0.7714,
  "full_path_accuracy": 0.4437,
  "macro_f1": 0.5876302081046875,
  "selection_metric": "macro_f1",
  "selection_point": "predict_100000_10000"
}
