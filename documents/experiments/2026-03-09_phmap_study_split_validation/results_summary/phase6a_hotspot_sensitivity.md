# PH-Map Phase 6A Hotspot Sensitivity

Hotspot parents were ranked from Phase 3 baseline `predict_100000_10000 + hierarchy_on` by `parent_correct_child_wrong_rate * n_cells`.

```json
{
  "reranker_top2": [
    "CD4+ T",
    "SMC de-differentiated"
  ],
  "reranker_top4": [
    "CD4+ T",
    "SMC de-differentiated",
    "Mph alveolar",
    "EC vascular"
  ],
  "reranker_top6": [
    "CD4+ T",
    "SMC de-differentiated",
    "Mph alveolar",
    "EC vascular",
    "Fibro adventitial",
    "CD8+ T"
  ]
}
```

| variant_name   | point                |   anno_lv4_macro_f1_mean |   anno_lv4_macro_f1_std |   anno_lv4_balanced_accuracy_mean |   anno_lv4_balanced_accuracy_std |   full_path_accuracy_mean |   full_path_accuracy_std |   anno_lv4_coverage_mean |   anno_lv4_coverage_std |   parent_correct_child_wrong_rate_mean |   parent_correct_child_wrong_rate_std |   path_break_rate_mean |   path_break_rate_std |   delta_vs_baseline_anno_lv4_macro_f1_mean |   delta_vs_baseline_full_path_accuracy_mean |   delta_vs_baseline_parent_correct_child_wrong_rate_mean |
|:---------------|:---------------------|-------------------------:|------------------------:|----------------------------------:|---------------------------------:|--------------------------:|-------------------------:|-------------------------:|------------------------:|---------------------------------------:|--------------------------------------:|-----------------------:|----------------------:|-------------------------------------------:|--------------------------------------------:|---------------------------------------------------------:|
| baseline       | predict_100000_10000 |                 0.587177 |              0.00569537 |                          0.586507 |                       0.00872936 |                   0.43836 |               0.0104196  |                  0.78234 |              0.0220404  |                                0.12348 |                            0.00986088 |                      0 |                     0 |                                0           |                                      0      |                                                  0       |
| reranker_top2  | predict_100000_10000 |                 0.589823 |              0.0042121  |                          0.595682 |                       0.00652727 |                   0.44906 |               0.00623242 |                  0.81754 |              0.0240889  |                                0.11272 |                            0.00582383 |                      0 |                     0 |                                0.00264644  |                                      0.0107 |                                                 -0.01076 |
| reranker_top4  | predict_100000_10000 |                 0.585025 |              0.00206501 |                          0.611945 |                       0.00512751 |                   0.46076 |               0.0054647  |                  0.8872  |              0.00868591 |                                0.10092 |                            0.00510461 |                      0 |                     0 |                               -0.00215183  |                                      0.0224 |                                                 -0.02256 |
| reranker_top6  | predict_100000_10000 |                 0.587388 |              0.00236824 |                          0.617886 |                       0.00477267 |                   0.46666 |               0.00492626 |                  0.89944 |              0.00889314 |                                0.09502 |                            0.00351525 |                      0 |                     0 |                                0.000211717 |                                      0.0283 |                                                 -0.02846 |