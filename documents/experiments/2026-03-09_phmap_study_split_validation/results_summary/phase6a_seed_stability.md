# PH-Map Phase 6A Seed Stability

- seeds: `101, 17, 2026, 23, 47`
- hotspot variants: `baseline, reranker_top2, reranker_top4, reranker_top6`

| variant_name   | point                |   anno_lv4_macro_f1_mean |   anno_lv4_macro_f1_std |   anno_lv4_balanced_accuracy_mean |   anno_lv4_balanced_accuracy_std |   full_path_accuracy_mean |   full_path_accuracy_std |   anno_lv4_coverage_mean |   anno_lv4_coverage_std |   parent_correct_child_wrong_rate_mean |   parent_correct_child_wrong_rate_std |   path_break_rate_mean |   path_break_rate_std |   delta_vs_baseline_anno_lv4_macro_f1_mean |   delta_vs_baseline_full_path_accuracy_mean |   delta_vs_baseline_parent_correct_child_wrong_rate_mean |
|:---------------|:---------------------|-------------------------:|------------------------:|----------------------------------:|---------------------------------:|--------------------------:|-------------------------:|-------------------------:|------------------------:|---------------------------------------:|--------------------------------------:|-----------------------:|----------------------:|-------------------------------------------:|--------------------------------------------:|---------------------------------------------------------:|
| baseline       | build_100000_eval10k |                 0.57067  |              0.0052976  |                          0.58414  |                       0.0044454  |                   0.43196 |               0.0103159  |                  0.78718 |              0.0246072  |                                0.13118 |                            0.00987076 |                      0 |                     0 |                                0           |                                     0       |                                                  0       |
| baseline       | predict_100000_10000 |                 0.587177 |              0.00569537 |                          0.586507 |                       0.00872936 |                   0.43836 |               0.0104196  |                  0.78234 |              0.0220404  |                                0.12348 |                            0.00986088 |                      0 |                     0 |                                0           |                                     0       |                                                  0       |
| reranker_top2  | build_100000_eval10k |                 0.573259 |              0.00216424 |                          0.592799 |                       0.00299414 |                   0.44304 |               0.00550618 |                  0.82432 |              0.0257106  |                                0.12    |                            0.00504777 |                      0 |                     0 |                                0.00258869  |                                     0.01108 |                                                 -0.01118 |
| reranker_top2  | predict_100000_10000 |                 0.589823 |              0.0042121  |                          0.595682 |                       0.00652727 |                   0.44906 |               0.00623242 |                  0.81754 |              0.0240889  |                                0.11272 |                            0.00582383 |                      0 |                     0 |                                0.00264644  |                                     0.0107  |                                                 -0.01076 |
| reranker_top4  | build_100000_eval10k |                 0.566579 |              0.00393659 |                          0.60495  |                       0.00299491 |                   0.45384 |               0.00496266 |                  0.89178 |              0.00694061 |                                0.1091  |                            0.00475079 |                      0 |                     0 |                               -0.00409071  |                                     0.02188 |                                                 -0.02208 |
| reranker_top4  | predict_100000_10000 |                 0.585025 |              0.00206501 |                          0.611945 |                       0.00512751 |                   0.46076 |               0.0054647  |                  0.8872  |              0.00868591 |                                0.10092 |                            0.00510461 |                      0 |                     0 |                               -0.00215183  |                                     0.0224  |                                                 -0.02256 |
| reranker_top6  | build_100000_eval10k |                 0.56956  |              0.00455597 |                          0.611636 |                       0.00330144 |                   0.46018 |               0.00389577 |                  0.90572 |              0.00686637 |                                0.10276 |                            0.00385071 |                      0 |                     0 |                               -0.00110971  |                                     0.02822 |                                                 -0.02842 |
| reranker_top6  | predict_100000_10000 |                 0.587388 |              0.00236824 |                          0.617886 |                       0.00477267 |                   0.46666 |               0.00492626 |                  0.89944 |              0.00889314 |                                0.09502 |                            0.00351525 |                      0 |                     0 |                                0.000211717 |                                     0.0283  |                                                 -0.02846 |

## Winner

```json
{
  "anno_lv4_balanced_accuracy_mean": 0.6178856850162057,
  "anno_lv4_balanced_accuracy_std": 0.0047726746494982765,
  "anno_lv4_coverage_mean": 0.89944,
  "anno_lv4_coverage_std": 0.008893143426258262,
  "anno_lv4_macro_f1_mean": 0.5873882447117136,
  "anno_lv4_macro_f1_std": 0.0023682437770089467,
  "delta_vs_baseline_anno_lv4_macro_f1_mean": 0.00021171674753195369,
  "delta_vs_baseline_full_path_accuracy_mean": 0.028300000000000002,
  "delta_vs_baseline_parent_correct_child_wrong_rate_mean": -0.028460000000000006,
  "full_path_accuracy_mean": 0.46665999999999996,
  "full_path_accuracy_std": 0.004926256184974548,
  "parent_correct_child_wrong_rate_mean": 0.09502,
  "parent_correct_child_wrong_rate_std": 0.0035152524802636864,
  "passes_gate": true,
  "path_break_rate_mean": 0.0,
  "path_break_rate_std": 0.0,
  "point": "predict_100000_10000",
  "variant_name": "reranker_top6"
}
```