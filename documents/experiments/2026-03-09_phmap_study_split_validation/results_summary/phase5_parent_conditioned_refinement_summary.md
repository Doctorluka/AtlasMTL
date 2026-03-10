# PH-Map Phase 5 Parent-Conditioned Child Refinement

- hotspot parents: `CD8+ T, Mph alveolar, Pericyte, CD4+ T`
- fitted rerankers: `CD4+ T, CD8+ T, Mph alveolar, Pericyte`

## Comparison

| variant_name                | point                | hierarchy_setting   |   anno_lv4_macro_f1 |   delta_vs_baseline_anno_lv4_macro_f1 |   anno_lv4_balanced_accuracy |   delta_vs_baseline_anno_lv4_balanced_accuracy |   full_path_accuracy |   delta_vs_baseline_full_path_accuracy |   anno_lv4_coverage |   delta_vs_baseline_anno_lv4_coverage |   parent_correct_child_wrong_rate |   delta_vs_baseline_parent_correct_child_wrong_rate |   path_break_rate |   delta_vs_baseline_path_break_rate |
|:----------------------------|:---------------------|:--------------------|--------------------:|--------------------------------------:|-----------------------------:|-----------------------------------------------:|---------------------:|---------------------------------------:|--------------------:|--------------------------------------:|----------------------------------:|----------------------------------------------------:|------------------:|------------------------------------:|
| baseline                    | build_100000_eval10k | on                  |            0.579605 |                            0          |                     0.589384 |                                      0         |               0.4408 |                                 0      |              0.845  |                                0      |                            0.12   |                                              0      |            0      |                              0      |
| parent_conditioned_reranker | build_100000_eval10k | on                  |            0.58494  |                            0.00533509 |                     0.609813 |                                      0.0204281 |               0.4586 |                                 0.0178 |              0.8902 |                                0.0452 |                            0.1022 |                                             -0.0178 |            0      |                              0      |
| baseline                    | build_100000_eval10k | off                 |            0.57895  |                            0          |                     0.615205 |                                      0         |               0.4408 |                                 0      |              0.9004 |                                0      |                            0.1214 |                                              0      |            0.0577 |                              0      |
| parent_conditioned_reranker | build_100000_eval10k | off                 |            0.589501 |                            0.0105518  |                     0.626921 |                                      0.0117155 |               0.4586 |                                 0.0178 |              0.921  |                                0.0206 |                            0.1034 |                                             -0.018  |            0.033  |                             -0.0247 |
| baseline                    | predict_100000_10000 | on                  |            0.591375 |                            0          |                     0.595755 |                                      0         |               0.4443 |                                 0      |              0.8404 |                                0      |                            0.1175 |                                              0      |            0      |                              0      |
| parent_conditioned_reranker | predict_100000_10000 | on                  |            0.602197 |                            0.0108218  |                     0.622339 |                                      0.0265836 |               0.463  |                                 0.0187 |              0.884  |                                0.0436 |                            0.0985 |                                             -0.019  |            0      |                              0      |
| baseline                    | predict_100000_10000 | off                 |            0.593351 |                            0          |                     0.613384 |                                      0         |               0.4443 |                                 0      |              0.893  |                                0      |                            0.1188 |                                              0      |            0.0557 |                              0      |
| parent_conditioned_reranker | predict_100000_10000 | off                 |            0.602937 |                            0.00958614 |                     0.632878 |                                      0.0194942 |               0.463  |                                 0.0187 |              0.9129 |                                0.0199 |                            0.0993 |                                             -0.0195 |            0.0312 |                             -0.0245 |

## Winner

{
  "anno_lv4_macro_f1": 0.6021965166599931,
  "delta_vs_baseline_anno_lv4_macro_f1": 0.010821751252037437,
  "delta_vs_baseline_full_path_accuracy": 0.01870000000000005,
  "full_path_accuracy": 0.463,
  "hierarchy_setting": "on",
  "point": "predict_100000_10000",
  "status": "selected",
  "winner_variant": "parent_conditioned_reranker"
}
