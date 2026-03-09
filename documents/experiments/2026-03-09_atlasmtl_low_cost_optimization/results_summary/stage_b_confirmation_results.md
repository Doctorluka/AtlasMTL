# Stage B Results

Rows: 32

| dataset                   | track    | point                | config_name       |   accuracy |   macro_f1 |   balanced_accuracy |   train_elapsed_seconds |   predict_elapsed_seconds | early_stopping_note   | status   |
|:--------------------------|:---------|:---------------------|:------------------|-----------:|-----------:|--------------------:|------------------------:|--------------------------:|:----------------------|:---------|
| DISCO_hPBMCs              | cpu_core | build_100000_eval10k | baseline          |     0.893  |   0.818682 |            0.80599  |                 40.1394 |                    0.0959 | stopped_early         | success  |
| DISCO_hPBMCs              | cpu_core | build_100000_eval10k | candidate_default |     0.8865 |   0.790599 |            0.782792 |                 30.4356 |                    0.1305 | stopped_early         | success  |
| DISCO_hPBMCs              | cpu_core | predict_100000_10000 | baseline          |     0.8871 |   0.795388 |            0.786349 |                 29.8993 |                    0.1246 | stopped_early         | success  |
| DISCO_hPBMCs              | cpu_core | predict_100000_10000 | candidate_default |     0.8906 |   0.804215 |            0.795134 |                 34.4684 |                    0.0958 | stopped_early         | success  |
| DISCO_hPBMCs              | gpu      | build_100000_eval10k | baseline          |     0.8899 |   0.790352 |            0.779853 |                  9.4917 |                    0.0548 | stopped_early         | success  |
| DISCO_hPBMCs              | gpu      | build_100000_eval10k | candidate_default |     0.8882 |   0.82485  |            0.808699 |                  8.2312 |                    0.0548 | stopped_early         | success  |
| DISCO_hPBMCs              | gpu      | predict_100000_10000 | baseline          |     0.8871 |   0.804038 |            0.791299 |                  8.2909 |                    0.0553 | stopped_early         | success  |
| DISCO_hPBMCs              | gpu      | predict_100000_10000 | candidate_default |     0.8865 |   0.845933 |            0.830666 |                  9.6851 |                    0.0552 | stopped_early         | success  |
| HLCA_Core                 | cpu_core | build_100000_eval10k | baseline          |     0.8744 |   0.781643 |            0.767551 |                 37.2105 |                    0.1167 | stopped_early         | success  |
| HLCA_Core                 | cpu_core | build_100000_eval10k | candidate_default |     0.8718 |   0.758765 |            0.736744 |                 34.1194 |                    0.1244 | stopped_early         | success  |
| HLCA_Core                 | cpu_core | predict_100000_10000 | baseline          |     0.8729 |   0.766291 |            0.753637 |                 33.9672 |                    0.0968 | stopped_early         | success  |
| HLCA_Core                 | cpu_core | predict_100000_10000 | candidate_default |     0.8726 |   0.752364 |            0.749309 |                 30.7538 |                    0.1135 | stopped_early         | success  |
| HLCA_Core                 | gpu      | build_100000_eval10k | baseline          |     0.872  |   0.748312 |            0.73363  |                  8.3232 |                    0.0551 | stopped_early         | success  |
| HLCA_Core                 | gpu      | build_100000_eval10k | candidate_default |     0.8673 |   0.762931 |            0.749662 |                  8.2276 |                    0.0551 | stopped_early         | success  |
| HLCA_Core                 | gpu      | predict_100000_10000 | baseline          |     0.8795 |   0.775684 |            0.764105 |                  9.5227 |                    0.0553 | stopped_early         | success  |
| HLCA_Core                 | gpu      | predict_100000_10000 | candidate_default |     0.8716 |   0.781362 |            0.765682 |                  9.567  |                    0.0633 | stopped_early         | success  |
| PHMap_Lung_Full_v43_light | cpu_core | build_100000_eval10k | baseline          |     0.5862 |   0.628379 |            0.632464 |                 33.8741 |                    0.0998 | stopped_early         | success  |
| PHMap_Lung_Full_v43_light | cpu_core | build_100000_eval10k | candidate_default |     0.589  |   0.626615 |            0.642404 |                 40.8036 |                    0.1179 | stopped_early         | success  |
| PHMap_Lung_Full_v43_light | cpu_core | predict_100000_10000 | baseline          |     0.5776 |   0.64339  |            0.665166 |                 37.3159 |                    0.1206 | stopped_early         | success  |
| PHMap_Lung_Full_v43_light | cpu_core | predict_100000_10000 | candidate_default |     0.573  |   0.644083 |            0.665604 |                 34.4298 |                    0.1189 | stopped_early         | success  |
| PHMap_Lung_Full_v43_light | gpu      | build_100000_eval10k | baseline          |     0.5885 |   0.638313 |            0.648263 |                  8.2459 |                    0.0554 | stopped_early         | success  |
| PHMap_Lung_Full_v43_light | gpu      | build_100000_eval10k | candidate_default |     0.5949 |   0.655267 |            0.667276 |                  9.5654 |                    0.0558 | stopped_early         | success  |
| PHMap_Lung_Full_v43_light | gpu      | predict_100000_10000 | baseline          |     0.5782 |   0.662659 |            0.677651 |                  8.2656 |                    0.0745 | stopped_early         | success  |
| PHMap_Lung_Full_v43_light | gpu      | predict_100000_10000 | candidate_default |     0.5815 |   0.653983 |            0.668898 |                  8.2419 |                    0.08   | stopped_early         | success  |
| mTCA                      | cpu_core | build_100000_eval10k | baseline          |     0.9406 |   0.854363 |            0.853049 |                 33.8141 |                    0.1242 | stopped_early         | success  |
| mTCA                      | cpu_core | build_100000_eval10k | candidate_default |     0.9415 |   0.848437 |            0.84916  |                 27.7985 |                    0.1168 | stopped_early         | success  |
| mTCA                      | cpu_core | predict_100000_10000 | baseline          |     0.942  |   0.837454 |            0.83585  |                 26.587  |                    0.126  | stopped_early         | success  |
| mTCA                      | cpu_core | predict_100000_10000 | candidate_default |     0.9448 |   0.877143 |            0.874252 |                 30.082  |                    0.0922 | stopped_early         | success  |
| mTCA                      | gpu      | build_100000_eval10k | baseline          |     0.937  |   0.841962 |            0.844369 |                  7.1613 |                    0.0623 | stopped_early         | success  |
| mTCA                      | gpu      | build_100000_eval10k | candidate_default |     0.9344 |   0.8563   |            0.851756 |                  9.5646 |                    0.0552 | stopped_early         | success  |
| mTCA                      | gpu      | predict_100000_10000 | baseline          |     0.9402 |   0.829994 |            0.834166 |                  7.1234 |                    0.0798 | stopped_early         | success  |
| mTCA                      | gpu      | predict_100000_10000 | candidate_default |     0.9421 |   0.851815 |            0.849796 |                  8.205  |                    0.0548 | stopped_early         | success  |
