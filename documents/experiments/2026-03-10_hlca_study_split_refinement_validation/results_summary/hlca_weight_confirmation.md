# HLCA Weight Confirmation

- evaluated configs: `3`
- evaluation rows: `6`

## Finest-Level Comparison

| config_name   | point                |   macro_f1 |   balanced_accuracy |   accuracy |   coverage |   unknown_rate |   full_path_accuracy |
|:--------------|:---------------------|-----------:|--------------------:|-----------:|-----------:|---------------:|---------------------:|
| mild_lv5      | build_100000_eval10k |   0.646105 |            0.625683 |     0.8377 |     0.9723 |         0.0277 |               0.8291 |
| strong_lv5    | build_100000_eval10k |   0.635175 |            0.620168 |     0.8395 |     0.9671 |         0.0329 |               0.8298 |
| uniform       | build_100000_eval10k |   0.650993 |            0.638436 |     0.8387 |     0.969  |         0.031  |               0.831  |
| mild_lv5      | predict_100000_10000 |   0.687933 |            0.674774 |     0.8285 |     0.9703 |         0.0297 |               0.819  |
| strong_lv5    | predict_100000_10000 |   0.675531 |            0.662135 |     0.8295 |     0.9636 |         0.0364 |               0.8179 |
| uniform       | predict_100000_10000 |   0.688732 |            0.682491 |     0.8336 |     0.9678 |         0.0322 |               0.8239 |

## Selected Base Config

{
  "balanced_accuracy": 0.6824906881230501,
  "best_config_name": "uniform",
  "coverage": 0.9678,
  "full_path_accuracy": 0.8239,
  "macro_f1": 0.6887323609459964,
  "selection_point": "predict_100000_10000",
  "selection_rationale": "highest_macro_f1",
  "unknown_rate": 0.0322
}
