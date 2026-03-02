# Results

## Main benchmark (no-obsm query)

| geometry_mode | knn_variant | accuracy_lv4 | covered_acc_lv4 | coverage_lv4 | unknown_rate_lv4 | macro_f1_lv4 | balanced_acc_lv4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| predicted_scanvi_head | knn_off | 0.7683 | 0.8294 | 0.9263 | 0.0737 | 0.6913 | 0.6589 |
| predicted_scanvi_head | knn_lowconf | 0.6557 | 0.8673 | 0.7560 | 0.2440 | 0.6096 | 0.5281 |
| predicted_scanvi_head | knn_all | 0.0603 | 0.0810 | 0.7447 | 0.2553 | 0.0141 | 0.0154 |
| latent_internal | knn_off | 0.7657 | 0.8201 | 0.9337 | 0.0663 | 0.6728 | 0.6387 |
| latent_internal | knn_lowconf | 0.7527 | 0.8492 | 0.8863 | 0.1137 | 0.6797 | 0.6324 |
| latent_internal | knn_all | 0.7533 | 0.8458 | 0.8907 | 0.1093 | 0.6848 | 0.6414 |

## Coordinate diagnostic (A only)

- `scanvi_continuity = 0.9710`
- `scanvi_trustworthiness = 0.9737`
- `scanvi_rmse = 1.3352`
- `scanvi_neighbor_overlap = 0.2153`

## Discussion notes

- Key tradeoff: `latent_internal` KNN improves covered accuracy but reduces
  coverage and does not beat `knn_off` in end-to-end accuracy.
- Geometry clearly matters: `predicted_scanvi_head` KNN is highly harmful,
  especially under `knn_all`, while `latent_internal` remains relatively stable.
- Next ablations: keep `latent_internal` as the main KNN branch and test vote
  mode, thresholds, prototype/full reference mode, and hierarchy interaction.
