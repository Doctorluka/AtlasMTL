# Multi-Level Benchmark Summary

- level-wise rows: `56`
- hierarchy rows: `16`
- reliability rows: `16`

## Mean Macro-F1 By Level

| dataset                   | track    | level            |   macro_f1 |
|:--------------------------|:---------|:-----------------|-----------:|
| DISCO_hPBMCs              | cpu_core | cell_subtype     |   0.793686 |
| DISCO_hPBMCs              | cpu_core | cell_type        |   0.815695 |
| DISCO_hPBMCs              | gpu      | cell_subtype     |   0.815602 |
| DISCO_hPBMCs              | gpu      | cell_type        |   0.85276  |
| HLCA_Core                 | cpu_core | ann_level_1      |   0.993662 |
| HLCA_Core                 | cpu_core | ann_level_2      |   0.867569 |
| HLCA_Core                 | cpu_core | ann_level_3      |   0.811202 |
| HLCA_Core                 | cpu_core | ann_level_4      |   0.784438 |
| HLCA_Core                 | cpu_core | ann_level_5      |   0.746021 |
| HLCA_Core                 | gpu      | ann_level_1      |   0.993931 |
| HLCA_Core                 | gpu      | ann_level_2      |   0.906767 |
| HLCA_Core                 | gpu      | ann_level_3      |   0.811662 |
| HLCA_Core                 | gpu      | ann_level_4      |   0.794702 |
| HLCA_Core                 | gpu      | ann_level_5      |   0.763194 |
| PHMap_Lung_Full_v43_light | cpu_core | anno_lv1         |   0.911082 |
| PHMap_Lung_Full_v43_light | cpu_core | anno_lv2         |   0.776103 |
| PHMap_Lung_Full_v43_light | cpu_core | anno_lv3         |   0.723212 |
| PHMap_Lung_Full_v43_light | cpu_core | anno_lv4         |   0.617775 |
| PHMap_Lung_Full_v43_light | gpu      | anno_lv1         |   0.911056 |
| PHMap_Lung_Full_v43_light | gpu      | anno_lv2         |   0.771955 |
| PHMap_Lung_Full_v43_light | gpu      | anno_lv3         |   0.719031 |
| PHMap_Lung_Full_v43_light | gpu      | anno_lv4         |   0.63319  |
| mTCA                      | cpu_core | Cell_type_level1 |   0.988131 |
| mTCA                      | cpu_core | Cell_type_level2 |   0.911064 |
| mTCA                      | cpu_core | Cell_type_level3 |   0.860687 |
| mTCA                      | gpu      | Cell_type_level1 |   0.98591  |
| mTCA                      | gpu      | Cell_type_level2 |   0.898512 |
| mTCA                      | gpu      | Cell_type_level3 |   0.844621 |

## Hierarchy Summary

| dataset                   | track    | point                |   full_path_accuracy |   full_path_coverage |   full_path_covered_accuracy |   mean_path_consistency_rate |   min_path_consistency_rate |
|:--------------------------|:---------|:---------------------|---------------------:|---------------------:|-----------------------------:|-----------------------------:|----------------------------:|
| DISCO_hPBMCs              | cpu_core | build_100000_eval10k |               0.8859 |               0.9845 |                     0.899848 |                            1 |                           1 |
| DISCO_hPBMCs              | cpu_core | predict_100000_10000 |               0.8839 |               0.9828 |                     0.899369 |                            1 |                           1 |
| DISCO_hPBMCs              | gpu      | build_100000_eval10k |               0.8868 |               0.986  |                     0.899391 |                            1 |                           1 |
| DISCO_hPBMCs              | gpu      | predict_100000_10000 |               0.8826 |               0.9847 |                     0.896314 |                            1 |                           1 |
| HLCA_Core                 | cpu_core | build_100000_eval10k |               0.8555 |               0.9739 |                     0.878427 |                            1 |                           1 |
| HLCA_Core                 | cpu_core | predict_100000_10000 |               0.861  |               0.9727 |                     0.885165 |                            1 |                           1 |
| HLCA_Core                 | gpu      | build_100000_eval10k |               0.864  |               0.9769 |                     0.88443  |                            1 |                           1 |
| HLCA_Core                 | gpu      | predict_100000_10000 |               0.8554 |               0.9706 |                     0.881311 |                            1 |                           1 |
| PHMap_Lung_Full_v43_light | cpu_core | build_100000_eval10k |               0.5732 |               0.7795 |                     0.735343 |                            1 |                           1 |
| PHMap_Lung_Full_v43_light | cpu_core | predict_100000_10000 |               0.5558 |               0.7574 |                     0.733826 |                            1 |                           1 |
| PHMap_Lung_Full_v43_light | gpu      | build_100000_eval10k |               0.5742 |               0.8043 |                     0.713913 |                            1 |                           1 |
| PHMap_Lung_Full_v43_light | gpu      | predict_100000_10000 |               0.5493 |               0.8318 |                     0.660375 |                            1 |                           1 |
| mTCA                      | cpu_core | build_100000_eval10k |               0.9365 |               0.9862 |                     0.949605 |                            1 |                           1 |
| mTCA                      | cpu_core | predict_100000_10000 |               0.9423 |               0.9888 |                     0.952973 |                            1 |                           1 |
| mTCA                      | gpu      | build_100000_eval10k |               0.9328 |               0.9908 |                     0.941461 |                            1 |                           1 |
| mTCA                      | gpu      | predict_100000_10000 |               0.9334 |               0.9871 |                     0.945598 |                            1 |                           1 |

## Finest-Level Reliability

| dataset                   | track    | point                | finest_level     |   coverage |   reject_rate |   covered_accuracy |      risk |        ece |       aurc |
|:--------------------------|:---------|:---------------------|:-----------------|-----------:|--------------:|-------------------:|----------:|-----------:|-----------:|
| DISCO_hPBMCs              | cpu_core | build_100000_eval10k | cell_subtype     |     0.9846 |        0.0154 |           0.899858 | 0.100142  | 0.019089   | 0.0225257  |
| DISCO_hPBMCs              | cpu_core | predict_100000_10000 | cell_subtype     |     0.9833 |        0.0167 |           0.898912 | 0.101088  | 0.0236026  | 0.022758   |
| DISCO_hPBMCs              | gpu      | build_100000_eval10k | cell_subtype     |     0.9861 |        0.0139 |           0.899402 | 0.100598  | 0.028225   | 0.023416   |
| DISCO_hPBMCs              | gpu      | predict_100000_10000 | cell_subtype     |     0.9851 |        0.0149 |           0.896051 | 0.103949  | 0.0336061  | 0.0230123  |
| HLCA_Core                 | cpu_core | build_100000_eval10k | ann_level_5      |     0.9833 |        0.0167 |           0.875623 | 0.124377  | 0.0574436  | 0.0296253  |
| HLCA_Core                 | cpu_core | predict_100000_10000 | ann_level_5      |     0.9834 |        0.0166 |           0.88133  | 0.11867   | 0.0444421  | 0.0268562  |
| HLCA_Core                 | gpu      | build_100000_eval10k | ann_level_5      |     0.9861 |        0.0139 |           0.881047 | 0.118953  | 0.05549    | 0.0273364  |
| HLCA_Core                 | gpu      | predict_100000_10000 | ann_level_5      |     0.9802 |        0.0198 |           0.878086 | 0.121914  | 0.0533037  | 0.0298625  |
| PHMap_Lung_Full_v43_light | cpu_core | build_100000_eval10k | anno_lv4         |     0.793  |        0.207  |           0.726986 | 0.273014  | 0.0791807  | 0.0992577  |
| PHMap_Lung_Full_v43_light | cpu_core | predict_100000_10000 | anno_lv4         |     0.7658 |        0.2342 |           0.731131 | 0.268869  | 0.0583758  | 0.0850619  |
| PHMap_Lung_Full_v43_light | gpu      | build_100000_eval10k | anno_lv4         |     0.8159 |        0.1841 |           0.707807 | 0.292193  | 0.110373   | 0.118107   |
| PHMap_Lung_Full_v43_light | gpu      | predict_100000_10000 | anno_lv4         |     0.8429 |        0.1571 |           0.654645 | 0.345355  | 0.14062    | 0.136849   |
| mTCA                      | cpu_core | build_100000_eval10k | Cell_type_level3 |     0.9877 |        0.0123 |           0.948871 | 0.0511289 | 0.0113969  | 0.00493351 |
| mTCA                      | cpu_core | predict_100000_10000 | Cell_type_level3 |     0.9903 |        0.0097 |           0.951833 | 0.0481672 | 0.011661   | 0.00488571 |
| mTCA                      | gpu      | build_100000_eval10k | Cell_type_level3 |     0.992  |        0.008  |           0.941028 | 0.0589718 | 0.019944   | 0.00574868 |
| mTCA                      | gpu      | predict_100000_10000 | Cell_type_level3 |     0.9888 |        0.0112 |           0.944782 | 0.0552184 | 0.00946337 | 0.00611021 |
