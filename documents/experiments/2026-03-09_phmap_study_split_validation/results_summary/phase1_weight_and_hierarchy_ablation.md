# PH-Map Study-Split Phase 1

- evaluation rows: `8`

## Finest-Level Rows

| config_name         | point                | hierarchy_setting   |   macro_f1 |   balanced_accuracy |   accuracy |   coverage |   unknown_rate |   covered_accuracy |
|:--------------------|:---------------------|:--------------------|-----------:|--------------------:|-----------:|-----------:|---------------:|-------------------:|
| lv4strong_candidate | build_100000_eval10k | off                 |   0.564177 |            0.576405 |     0.4835 |     0.8179 |         0.1821 |           0.591148 |
| lv4strong_candidate | build_100000_eval10k | on                  |   0.560272 |            0.561906 |     0.4685 |     0.7764 |         0.2236 |           0.603426 |
| lv4strong_candidate | predict_100000_10000 | off                 |   0.553722 |            0.561504 |     0.484  |     0.807  |         0.193  |           0.599752 |
| lv4strong_candidate | predict_100000_10000 | on                  |   0.552978 |            0.549828 |     0.4718 |     0.7668 |         0.2332 |           0.615284 |
| uniform_control     | build_100000_eval10k | off                 |   0.550164 |            0.557763 |     0.4757 |     0.7921 |         0.2079 |           0.600555 |
| uniform_control     | build_100000_eval10k | on                  |   0.549254 |            0.550158 |     0.4638 |     0.7663 |         0.2337 |           0.605246 |
| uniform_control     | predict_100000_10000 | off                 |   0.544579 |            0.544546 |     0.4736 |     0.7842 |         0.2158 |           0.603928 |
| uniform_control     | predict_100000_10000 | on                  |   0.541423 |            0.535524 |     0.4619 |     0.7574 |         0.2426 |           0.609849 |

## Weight Comparison

| point                | hierarchy_setting   |   macro_f1_uniform |   macro_f1_lv4strong |   delta_macro_f1 |   full_path_accuracy_uniform |   full_path_accuracy_lv4strong |   delta_full_path_accuracy |   coverage_uniform |   coverage_lv4strong |   delta_coverage |
|:---------------------|:--------------------|-------------------:|---------------------:|-----------------:|-----------------------------:|-------------------------------:|---------------------------:|-------------------:|---------------------:|-----------------:|
| build_100000_eval10k | off                 |           0.550164 |             0.564177 |       0.0140132  |                       0.4548 |                         0.4591 |                     0.0043 |             0.7921 |               0.8179 |           0.0258 |
| build_100000_eval10k | on                  |           0.549254 |             0.560272 |       0.0110175  |                       0.4548 |                         0.4591 |                     0.0043 |             0.7663 |               0.7764 |           0.0101 |
| predict_100000_10000 | off                 |           0.544579 |             0.553722 |       0.00914223 |                       0.457  |                         0.4639 |                     0.0069 |             0.7842 |               0.807  |           0.0228 |
| predict_100000_10000 | on                  |           0.541423 |             0.552978 |       0.0115546  |                       0.457  |                         0.4639 |                     0.0069 |             0.7574 |               0.7668 |           0.0094 |

## Hierarchy Tradeoff

| config_name         | point                |   macro_f1_on |   macro_f1_off |   delta_off_minus_on_macro_f1 |   coverage_on |   coverage_off |   delta_off_minus_on_coverage |   mean_path_consistency_rate_on |   mean_path_consistency_rate_off |
|:--------------------|:---------------------|--------------:|---------------:|------------------------------:|--------------:|---------------:|------------------------------:|--------------------------------:|---------------------------------:|
| lv4strong_candidate | build_100000_eval10k |      0.560272 |       0.564177 |                   0.00390498  |        0.7764 |         0.8179 |                        0.0415 |                               1 |                         0.975497 |
| lv4strong_candidate | predict_100000_10000 |      0.552978 |       0.553722 |                   0.000743533 |        0.7668 |         0.807  |                        0.0402 |                               1 |                         0.975509 |
| uniform_control     | build_100000_eval10k |      0.549254 |       0.550164 |                   0.000909362 |        0.7663 |         0.7921 |                        0.0258 |                               1 |                         0.983285 |
| uniform_control     | predict_100000_10000 |      0.541423 |       0.544579 |                   0.00315591  |        0.7574 |         0.7842 |                        0.0268 |                               1 |                         0.983413 |
