# PH-Map Study-Split Phase 3 Tradeoff Attribution

## Finest-Level Rows

| config_name                 | point                | hierarchy_setting   |   macro_f1 |   balanced_accuracy |   accuracy |   coverage |   unknown_rate |
|:----------------------------|:---------------------|:--------------------|-----------:|--------------------:|-----------:|-----------:|---------------:|
| lv4strong_baseline          | build_100000_eval10k | off                 |   0.574095 |            0.579192 |     0.4873 |     0.8177 |         0.1823 |
| lv4strong_baseline          | build_100000_eval10k | on                  |   0.566347 |            0.563701 |     0.4645 |     0.7765 |         0.2235 |
| lv4strong_baseline          | predict_100000_10000 | off                 |   0.565763 |            0.564985 |     0.4834 |     0.8117 |         0.1883 |
| lv4strong_baseline          | predict_100000_10000 | on                  |   0.560001 |            0.549865 |     0.4643 |     0.7734 |         0.2266 |
| lv4strong_plus_class_weight | build_100000_eval10k | off                 |   0.57895  |            0.615205 |     0.4708 |     0.9004 |         0.0996 |
| lv4strong_plus_class_weight | build_100000_eval10k | on                  |   0.579605 |            0.589384 |     0.4499 |     0.845  |         0.155  |
| lv4strong_plus_class_weight | predict_100000_10000 | off                 |   0.593351 |            0.613384 |     0.4709 |     0.893  |         0.107  |
| lv4strong_plus_class_weight | predict_100000_10000 | on                  |   0.591375 |            0.595755 |     0.4533 |     0.8404 |         0.1596 |

## Hierarchy Rows

| config_name                 | point                | hierarchy_setting   |   full_path_accuracy |   full_path_coverage |   mean_path_consistency_rate |
|:----------------------------|:---------------------|:--------------------|---------------------:|---------------------:|-----------------------------:|
| lv4strong_baseline          | build_100000_eval10k | off                 |               0.4558 |               0.8142 |                     0.977009 |
| lv4strong_baseline          | build_100000_eval10k | on                  |               0.4558 |               0.7619 |                     1        |
| lv4strong_baseline          | predict_100000_10000 | off                 |               0.4562 |               0.8077 |                     0.977662 |
| lv4strong_baseline          | predict_100000_10000 | on                  |               0.4562 |               0.7575 |                     1        |
| lv4strong_plus_class_weight | build_100000_eval10k | off                 |               0.4408 |               0.8937 |                     0.9729   |
| lv4strong_plus_class_weight | build_100000_eval10k | on                  |               0.4408 |               0.8273 |                     1        |
| lv4strong_plus_class_weight | predict_100000_10000 | off                 |               0.4443 |               0.8861 |                     0.973065 |
| lv4strong_plus_class_weight | predict_100000_10000 | on                  |               0.4443 |               0.822  |                     1        |

## Parent-Child Breakdown

| config_name                 | point                | hierarchy_setting   | child_col   | parent_col   |   path_consistency_rate |   parent_correct_child_wrong_rate |   parent_wrong_child_correct_rate |   both_wrong_rate |   path_break_rate |
|:----------------------------|:---------------------|:--------------------|:------------|:-------------|------------------------:|----------------------------------:|----------------------------------:|------------------:|------------------:|
| lv4strong_baseline          | build_100000_eval10k | off                 | anno_lv2    | anno_lv1     |                0.992732 |                            0.3267 |                            0.0039 |            0.0121 |            0.0072 |
| lv4strong_baseline          | build_100000_eval10k | off                 | anno_lv3    | anno_lv2     |                0.990195 |                            0.1024 |                            0.0063 |            0.3325 |            0.0096 |
| lv4strong_baseline          | build_100000_eval10k | off                 | anno_lv4    | anno_lv3     |                0.948098 |                            0.103  |                            0.0252 |            0.4097 |            0.0423 |
| lv4strong_baseline          | build_100000_eval10k | on                  | anno_lv2    | anno_lv1     |                1        |                            0.3267 |                            0.0002 |            0.0158 |            0      |
| lv4strong_baseline          | build_100000_eval10k | on                  | anno_lv3    | anno_lv2     |                1        |                            0.1013 |                            0.0037 |            0.3388 |            0      |
| lv4strong_baseline          | build_100000_eval10k | on                  | anno_lv4    | anno_lv3     |                1        |                            0.1014 |                            0.006  |            0.4341 |            0      |
| lv4strong_baseline          | predict_100000_10000 | off                 | anno_lv2    | anno_lv1     |                0.991824 |                            0.3344 |                            0.0048 |            0.0134 |            0.0081 |
| lv4strong_baseline          | predict_100000_10000 | off                 | anno_lv3    | anno_lv2     |                0.990619 |                            0.0937 |                            0.0053 |            0.3425 |            0.0092 |
| lv4strong_baseline          | predict_100000_10000 | off                 | anno_lv4    | anno_lv3     |                0.950544 |                            0.1014 |                            0.021  |            0.4152 |            0.04   |
| lv4strong_baseline          | predict_100000_10000 | on                  | anno_lv2    | anno_lv1     |                1        |                            0.3344 |                            0.0001 |            0.0181 |            0      |
| lv4strong_baseline          | predict_100000_10000 | on                  | anno_lv3    | anno_lv2     |                1        |                            0.093  |                            0.0058 |            0.3467 |            0      |
| lv4strong_baseline          | predict_100000_10000 | on                  | anno_lv4    | anno_lv3     |                1        |                            0.1003 |                            0.0043 |            0.4354 |            0      |
| lv4strong_plus_class_weight | build_100000_eval10k | off                 | anno_lv2    | anno_lv1     |                0.99284  |                            0.3223 |                            0.0046 |            0.0138 |            0.0071 |
| lv4strong_plus_class_weight | build_100000_eval10k | off                 | anno_lv3    | anno_lv2     |                0.990321 |                            0.1011 |                            0.0051 |            0.331  |            0.0095 |
| lv4strong_plus_class_weight | build_100000_eval10k | off                 | anno_lv4    | anno_lv3     |                0.935538 |                            0.1214 |                            0.0243 |            0.4078 |            0.0577 |
| lv4strong_plus_class_weight | build_100000_eval10k | on                  | anno_lv2    | anno_lv1     |                1        |                            0.3223 |                            0.0001 |            0.0183 |            0      |
| lv4strong_plus_class_weight | build_100000_eval10k | on                  | anno_lv3    | anno_lv2     |                1        |                            0.1006 |                            0.0055 |            0.3351 |            0      |
| lv4strong_plus_class_weight | build_100000_eval10k | on                  | anno_lv4    | anno_lv3     |                1        |                            0.12   |                            0.0056 |            0.4301 |            0      |
| lv4strong_plus_class_weight | predict_100000_10000 | off                 | anno_lv2    | anno_lv1     |                0.991536 |                            0.3312 |                            0.0054 |            0.0127 |            0.0084 |
| lv4strong_plus_class_weight | predict_100000_10000 | off                 | anno_lv3    | anno_lv2     |                0.99044  |                            0.0919 |                            0.0043 |            0.3396 |            0.0094 |
| lv4strong_plus_class_weight | predict_100000_10000 | off                 | anno_lv4    | anno_lv3     |                0.937218 |                            0.1188 |                            0.0212 |            0.4103 |            0.0557 |
| lv4strong_plus_class_weight | predict_100000_10000 | on                  | anno_lv2    | anno_lv1     |                1        |                            0.3312 |                            0.0002 |            0.0179 |            0      |
| lv4strong_plus_class_weight | predict_100000_10000 | on                  | anno_lv3    | anno_lv2     |                1        |                            0.0911 |                            0.0055 |            0.3436 |            0      |
| lv4strong_plus_class_weight | predict_100000_10000 | on                  | anno_lv4    | anno_lv3     |                1        |                            0.1175 |                            0.0055 |            0.4292 |            0      |

## Subtree Hotspots

| config_name        | point                | hierarchy_setting   | parent_label          |   n_cells |   lv4_accuracy |   lv4_unknown_rate |   parent_correct_child_wrong_rate |   path_break_rate |
|:-------------------|:---------------------|:--------------------|:----------------------|----------:|---------------:|-------------------:|----------------------------------:|------------------:|
| lv4strong_baseline | build_100000_eval10k | off                 | CD8+ T                |       463 |      0.650108  |          0.153348  |                         0.157667  |         0.155508  |
| lv4strong_baseline | build_100000_eval10k | off                 | NK CD56 high          |        71 |      0.633803  |          0.169014  |                         0.0140845 |         0.140845  |
| lv4strong_baseline | build_100000_eval10k | off                 | Pericyte              |       109 |      0.486239  |          0.165138  |                         0.0550459 |         0.100917  |
| lv4strong_baseline | build_100000_eval10k | off                 | SMC contractile       |       959 |      0.83316   |          0.0302398 |                         0.012513  |         0.0990615 |
| lv4strong_baseline | build_100000_eval10k | off                 | CD4+ T                |       712 |      0.429775  |          0.231742  |                         0.433989  |         0.0884831 |
| lv4strong_baseline | build_100000_eval10k | off                 | SMC de-differentiated |       813 |      0.570726  |          0.095941  |                         0.287823  |         0.0750308 |
| lv4strong_baseline | build_100000_eval10k | off                 | Mph alveolar          |       612 |      0.552288  |          0.0424837 |                         0.0751634 |         0.0669935 |
| lv4strong_baseline | build_100000_eval10k | off                 | Epi AT2               |       267 |      0.779026  |          0.0486891 |                         0.0262172 |         0.0524345 |
| lv4strong_baseline | build_100000_eval10k | off                 | Fibro subpleural      |        30 |      0.0333333 |          0.133333  |                         0         |         0.0333333 |
| lv4strong_baseline | build_100000_eval10k | off                 | Fibro alveolar        |       290 |      0.558621  |          0.2       |                         0.155172  |         0.0310345 |
| lv4strong_baseline | build_100000_eval10k | off                 | Epi AT1               |       292 |      0.934932  |          0.0171233 |                         0.0205479 |         0.0308219 |
| lv4strong_baseline | build_100000_eval10k | off                 | Epi Goblet            |        35 |      0.771429  |          0.114286  |                         0.0571429 |         0.0285714 |
| lv4strong_baseline | build_100000_eval10k | off                 | DC                    |        79 |      0.683544  |          0.126582  |                         0.0632911 |         0.0253165 |
| lv4strong_baseline | build_100000_eval10k | off                 | B                     |        41 |      0.926829  |          0.0487805 |                         0         |         0.0243902 |
| lv4strong_baseline | build_100000_eval10k | off                 | Plasma                |        45 |      0.977778  |          0.0222222 |                         0         |         0.0222222 |
| lv4strong_baseline | build_100000_eval10k | off                 | EC vascular           |       563 |      0.690941  |          0.110124  |                         0.271758  |         0.0142096 |

## Study Breakdown

| config_name                 | point                | hierarchy_setting   | study                |   anno_lv4_macro_f1 |   coverage |   unknown_rate |   full_path_accuracy |
|:----------------------------|:---------------------|:--------------------|:---------------------|--------------------:|-----------:|---------------:|---------------------:|
| lv4strong_baseline          | build_100000_eval10k | off                 | Jonas_Schupp_2021    |            0        |   0.600069 |      0.399931  |             0        |
| lv4strong_baseline          | build_100000_eval10k | off                 | Slaven_Crnkovic_2022 |            0.655537 |   0.940588 |      0.0594115 |             0.651075 |
| lv4strong_baseline          | build_100000_eval10k | off                 | Tijana_Tuhy_2025     |            0.492023 |   0.808427 |      0.191573  |             0.621348 |
| lv4strong_baseline          | build_100000_eval10k | on                  | Jonas_Schupp_2021    |            0        |   0.597327 |      0.402673  |             0        |
| lv4strong_baseline          | build_100000_eval10k | on                  | Slaven_Crnkovic_2022 |            0.645433 |   0.871935 |      0.128065  |             0.651075 |
| lv4strong_baseline          | build_100000_eval10k | on                  | Tijana_Tuhy_2025     |            0.487666 |   0.785955 |      0.214045  |             0.621348 |
| lv4strong_baseline          | predict_100000_10000 | off                 | Jonas_Schupp_2021    |            0        |   0.58623  |      0.41377   |             0        |
| lv4strong_baseline          | predict_100000_10000 | off                 | Slaven_Crnkovic_2022 |            0.664649 |   0.940533 |      0.0594667 |             0.661423 |
| lv4strong_baseline          | predict_100000_10000 | off                 | Tijana_Tuhy_2025     |            0.512745 |   0.817101 |      0.182899  |             0.630804 |
| lv4strong_baseline          | predict_100000_10000 | on                  | Jonas_Schupp_2021    |            0        |   0.582589 |      0.417411  |             0        |
| lv4strong_baseline          | predict_100000_10000 | on                  | Slaven_Crnkovic_2022 |            0.657518 |   0.878381 |      0.121619  |             0.661423 |
| lv4strong_baseline          | predict_100000_10000 | on                  | Tijana_Tuhy_2025     |            0.508718 |   0.789921 |      0.210079  |             0.630804 |
| lv4strong_plus_class_weight | build_100000_eval10k | off                 | Jonas_Schupp_2021    |            0        |   0.85024  |      0.14976   |             0        |
| lv4strong_plus_class_weight | build_100000_eval10k | off                 | Slaven_Crnkovic_2022 |            0.641568 |   0.950773 |      0.0492267 |             0.629762 |
| lv4strong_plus_class_weight | build_100000_eval10k | off                 | Tijana_Tuhy_2025     |            0.509617 |   0.832584 |      0.167416  |             0.600562 |
| lv4strong_plus_class_weight | build_100000_eval10k | on                  | Jonas_Schupp_2021    |            0        |   0.840302 |      0.159698  |             0        |
| lv4strong_plus_class_weight | build_100000_eval10k | on                  | Slaven_Crnkovic_2022 |            0.655462 |   0.870238 |      0.129762  |             0.629762 |
| lv4strong_plus_class_weight | build_100000_eval10k | on                  | Tijana_Tuhy_2025     |            0.497562 |   0.777528 |      0.222472  |             0.600562 |
| lv4strong_plus_class_weight | predict_100000_10000 | off                 | Jonas_Schupp_2021    |            0        |   0.817941 |      0.182059  |             0        |
| lv4strong_plus_class_weight | predict_100000_10000 | off                 | Slaven_Crnkovic_2022 |            0.654984 |   0.951467 |      0.0485325 |             0.640706 |
| lv4strong_plus_class_weight | predict_100000_10000 | off                 | Tijana_Tuhy_2025     |            0.561603 |   0.848811 |      0.151189  |             0.624575 |
| lv4strong_plus_class_weight | predict_100000_10000 | on                  | Jonas_Schupp_2021    |            0        |   0.811983 |      0.188017  |             0        |
| lv4strong_plus_class_weight | predict_100000_10000 | on                  | Slaven_Crnkovic_2022 |            0.651448 |   0.873777 |      0.126223  |             0.640706 |
| lv4strong_plus_class_weight | predict_100000_10000 | on                  | Tijana_Tuhy_2025     |            0.56147  |   0.790487 |      0.209513  |             0.624575 |
