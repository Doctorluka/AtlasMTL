# Phase 1 PHMap Weight And Hierarchy Ablation

- evaluation rows: `8`
- finest-level rows: `8`

## Finest-Level Rows

| config_name         | point                | hierarchy_setting   |   macro_f1 |   balanced_accuracy |   accuracy |   coverage |   unknown_rate |   covered_accuracy |
|:--------------------|:---------------------|:--------------------|-----------:|--------------------:|-----------:|-----------:|---------------:|-------------------:|
| lv4strong_candidate | build_100000_eval10k | off                 |   0.629506 |            0.642747 |     0.5896 |     0.8948 |         0.1052 |           0.658918 |
| lv4strong_candidate | build_100000_eval10k | on                  |   0.629088 |            0.633238 |     0.5833 |     0.8681 |         0.1319 |           0.671927 |
| lv4strong_candidate | predict_100000_10000 | off                 |   0.649373 |            0.672428 |     0.5801 |     0.8899 |         0.1101 |           0.651871 |
| lv4strong_candidate | predict_100000_10000 | on                  |   0.648124 |            0.661315 |     0.5712 |     0.8622 |         0.1378 |           0.662491 |
| uniform_control     | build_100000_eval10k | off                 |   0.625784 |            0.637851 |     0.5816 |     0.8972 |         0.1028 |           0.648239 |
| uniform_control     | build_100000_eval10k | on                  |   0.624404 |            0.629118 |     0.5713 |     0.8754 |         0.1246 |           0.652616 |
| uniform_control     | predict_100000_10000 | off                 |   0.643599 |            0.658016 |     0.5732 |     0.8923 |         0.1077 |           0.642385 |
| uniform_control     | predict_100000_10000 | on                  |   0.639902 |            0.647408 |     0.5598 |     0.8686 |         0.1314 |           0.644485 |

## Weight Comparison

| point                | hierarchy_setting   |   macro_f1_uniform |   macro_f1_lv4strong |   delta_macro_f1 |   full_path_accuracy_uniform |   full_path_accuracy_lv4strong |   delta_full_path_accuracy |   coverage_uniform |   coverage_lv4strong |   delta_coverage |
|:---------------------|:--------------------|-------------------:|---------------------:|-----------------:|-----------------------------:|-------------------------------:|---------------------------:|-------------------:|---------------------:|-----------------:|
| build_100000_eval10k | off                 |           0.625784 |             0.629506 |       0.00372204 |                       0.5675 |                         0.5777 |                     0.0102 |             0.8972 |               0.8948 |          -0.0024 |
| build_100000_eval10k | on                  |           0.624404 |             0.629088 |       0.00468469 |                       0.5675 |                         0.5777 |                     0.0102 |             0.8754 |               0.8681 |          -0.0073 |
| predict_100000_10000 | off                 |           0.643599 |             0.649373 |       0.00577355 |                       0.5557 |                         0.5667 |                     0.011  |             0.8923 |               0.8899 |          -0.0024 |
| predict_100000_10000 | on                  |           0.639902 |             0.648124 |       0.00822243 |                       0.5557 |                         0.5667 |                     0.011  |             0.8686 |               0.8622 |          -0.0064 |

## Hierarchy Tradeoff

| config_name         | point                |   macro_f1_on |   macro_f1_off |   delta_off_minus_on_macro_f1 |   coverage_on |   coverage_off |   delta_off_minus_on_coverage |   unknown_rate_on |   unknown_rate_off |   delta_off_minus_on_unknown_rate |   mean_path_consistency_rate_on |   mean_path_consistency_rate_off |
|:--------------------|:---------------------|--------------:|---------------:|------------------------------:|--------------:|---------------:|------------------------------:|------------------:|-------------------:|----------------------------------:|--------------------------------:|---------------------------------:|
| lv4strong_candidate | build_100000_eval10k |      0.629088 |       0.629506 |                   0.000417994 |        0.8681 |         0.8948 |                        0.0267 |            0.1319 |             0.1052 |                           -0.0267 |                               1 |                         0.985537 |
| lv4strong_candidate | predict_100000_10000 |      0.648124 |       0.649373 |                   0.00124853  |        0.8622 |         0.8899 |                        0.0277 |            0.1378 |             0.1101 |                           -0.0277 |                               1 |                         0.985008 |
| uniform_control     | build_100000_eval10k |      0.624404 |       0.625784 |                   0.00138065  |        0.8754 |         0.8972 |                        0.0218 |            0.1246 |             0.1028 |                           -0.0218 |                               1 |                         0.988155 |
| uniform_control     | predict_100000_10000 |      0.639902 |       0.643599 |                   0.00369742  |        0.8686 |         0.8923 |                        0.0237 |            0.1314 |             0.1077 |                           -0.0237 |                               1 |                         0.987445 |
