# Formal Refresh Results

Rows: 20

Summary:

- this fifth-round formal refresh does not justify replacing the retained
  third-wave manuscript-grade AtlasMTL baseline rows
- the refreshed configuration (`AdamW + weight_decay=5e-5`) remains the
  software default
- main-panel evidence is insufficient for manuscript-table replacement because
  mean `delta_macro_f1 = -0.005250`, GPU headline improvements are only `4/8`,
  and `DISCO_hPBMCs / gpu / predict_100000_10000` regressed by `-0.054224`

| dataset                   | track    | point                | scope         |   old_macro_f1 |   new_macro_f1 |   delta_macro_f1 |   old_train_elapsed_seconds |   new_train_elapsed_seconds |   delta_train_elapsed_seconds | status   |
|:--------------------------|:---------|:---------------------|:--------------|---------------:|---------------:|-----------------:|----------------------------:|----------------------------:|------------------------------:|:---------|
| DISCO_hPBMCs              | cpu_core | build_100000_eval10k | main          |       0.78269  |       0.797111 |      0.0144211   |                     27.1112 |                     38.0075 |                       10.8963 | success  |
| DISCO_hPBMCs              | cpu_core | predict_100000_10000 | main          |       0.816976 |       0.811014 |     -0.00596204  |                     33.9543 |                     38.3708 |                        4.4165 | success  |
| DISCO_hPBMCs              | gpu      | build_100000_eval10k | main          |       0.796364 |       0.802549 |      0.006185    |                      9.5788 |                      8.1509 |                       -1.4279 | success  |
| DISCO_hPBMCs              | gpu      | predict_100000_10000 | main          |       0.845199 |       0.790975 |     -0.0542236   |                      9.4562 |                      7.3796 |                       -2.0766 | success  |
| HLCA_Core                 | cpu_core | build_100000_eval10k | main          |       0.771325 |       0.763056 |     -0.00826915  |                     33.7811 |                     33.4657 |                       -0.3154 | success  |
| HLCA_Core                 | cpu_core | predict_100000_10000 | main          |       0.75782  |       0.757266 |     -0.000553593 |                     36.6427 |                     33.7949 |                       -2.8478 | success  |
| HLCA_Core                 | gpu      | build_100000_eval10k | main          |       0.761623 |       0.756436 |     -0.00518681  |                      8.416  |                      8.2404 |                       -0.1756 | success  |
| HLCA_Core                 | gpu      | predict_100000_10000 | main          |       0.753877 |       0.768177 |      0.0142999   |                      7.9905 |                      8.3018 |                        0.3113 | success  |
| PHMap_Lung_Full_v43_light | cpu_core | build_100000_eval10k | main          |       0.62683  |       0.624332 |     -0.00249754  |                     37.6296 |                     37.1321 |                       -0.4975 | success  |
| PHMap_Lung_Full_v43_light | cpu_core | predict_100000_10000 | main          |       0.649816 |       0.644464 |     -0.00535224  |                     36.6934 |                     34.0993 |                       -2.5941 | success  |
| PHMap_Lung_Full_v43_light | gpu      | build_100000_eval10k | main          |       0.651011 |       0.632335 |     -0.0186755   |                      9.6157 |                      8.3429 |                       -1.2728 | success  |
| PHMap_Lung_Full_v43_light | gpu      | predict_100000_10000 | main          |       0.650565 |       0.655592 |      0.00502656  |                      9.5201 |                      8.126  |                       -1.3941 | success  |
| Vento                     | cpu_core | build_50000_eval10k  | supplementary |       0.896494 |       0.910743 |      0.0142491   |                     14.1481 |                     17.1043 |                        2.9562 | success  |
| Vento                     | cpu_core | predict_50000_10000  | supplementary |       0.90559  |       0.908488 |      0.00289808  |                     17.5694 |                     20.1594 |                        2.59   | success  |
| Vento                     | gpu      | build_50000_eval10k  | supplementary |       0.913253 |       0.916553 |      0.00329933  |                      4.0698 |                      4.1375 |                        0.0677 | success  |
| Vento                     | gpu      | predict_50000_10000  | supplementary |       0.908031 |       0.876129 |     -0.0319015   |                      4.6618 |                      4.0795 |                       -0.5823 | success  |
| mTCA                      | cpu_core | build_100000_eval10k | main          |       0.840302 |       0.844742 |      0.00444039  |                     27.0559 |                     27.3697 |                        0.3138 | success  |
| mTCA                      | cpu_core | predict_100000_10000 | main          |       0.874091 |       0.864747 |     -0.00934331  |                     33.3265 |                     27.1909 |                       -6.1356 | success  |
| mTCA                      | gpu      | build_100000_eval10k | main          |       0.841207 |       0.843516 |      0.00230955  |                      8.1451 |                      9.535  |                        1.3899 | success  |
| mTCA                      | gpu      | predict_100000_10000 | main          |       0.852258 |       0.831641 |     -0.0206171   |                      8.2364 |                      8.1743 |                       -0.0621 | success  |
