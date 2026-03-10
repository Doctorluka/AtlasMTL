# Sixth-Round Expert Consultation Summary

## Scope

This note summarizes the latest sixth-round multi-level annotation results for
expert discussion.

Included:

- the completed sixth-round `v1` multi-level benchmark
- completed sixth-round follow-up weight probes on `PHMap` and `HLCA`

Excluded:

- the interrupted `HLCA full matrix` probe

The authoritative run-level source files remain:

- [levelwise_performance.csv](/home/data/fhz/project/phmap_package/atlasmtl/documents/experiments/2026-03-09_multilevel_annotation_benchmark/results_summary/levelwise_performance.csv)
- [hierarchy_performance.csv](/home/data/fhz/project/phmap_package/atlasmtl/documents/experiments/2026-03-09_multilevel_annotation_benchmark/results_summary/hierarchy_performance.csv)
- [reliability_performance.csv](/home/data/fhz/project/phmap_package/atlasmtl/documents/experiments/2026-03-09_multilevel_annotation_benchmark/results_summary/reliability_performance.csv)
- [multilevel_experiment_report.md](/home/data/fhz/project/phmap_package/atlasmtl/documents/experiments/2026-03-09_multilevel_annotation_benchmark/results_summary/multilevel_experiment_report.md)
- [hierarchy_aware_discussion_note.md](/home/data/fhz/project/phmap_package/atlasmtl/documents/experiments/2026-03-09_multilevel_annotation_benchmark/results_summary/hierarchy_aware_discussion_note.md)

## Protocol

### Core design

The sixth round was designed as an `AtlasMTL-only` multi-level annotation
benchmark rather than a new all-method formal comparison round.

Fixed settings:

- `knn_correction="off"`
- `input_transform="binary"`
- `optimizer_name="adamw"`
- `weight_decay=5e-5`
- `scheduler_name=None`
- `enforce_hierarchy=True`

Execution matrix:

- datasets: `HLCA_Core`, `PHMap_Lung_Full_v43_light`, `DISCO_hPBMCs`, `mTCA`
- tracks: `cpu_core`, `gpu`
- points:
  - `build_100000_eval10k`
  - `predict_100000_10000`
- total: `4 x 2 x 2 = 16 runs`

### Dataset hierarchy inventory

| dataset | levels | coarsest | finest | classes by level |
| ------- | ------ | -------- | ------ | ---------------- |
| `HLCA_Core` | 5 | `ann_level_1` | `ann_level_5` | `4, 11, 27, 51, 61` |
| `PHMap_Lung_Full_v43_light` | 4 | `anno_lv1` | `anno_lv4` | `6, 19, 30, 55` |
| `DISCO_hPBMCs` | 2 | `cell_type` | `cell_subtype` | `14, 24` |
| `mTCA` | 3 | `Cell_type_level1` | `Cell_type_level3` | `3, 16, 33` |

### Feature-space contract

The sixth round inherited the same prepared input contract as the formal
third-wave scaling round.

Important clarification:

- the sixth-round manifests did not rerun HVG selection explicitly
- however, the prepared inputs inherited from the third-wave preparation chain
  were built from prep manifests that fixed `feature_space: hvg` and
  `n_top_genes: 3000`
- so the effective feature-space contract for the sixth-round `v1` runs was
  `HVG 3000`

### Runtime caveat

All `16/16` runs completed successfully.

- CPU: `8/8` success, but all `8/8` carried
  `runtime_fairness_degraded=True`
- GPU: `8/8` success, no degraded-runtime flags

The CPU degraded flag was caused by restricted-environment `joblib` serial
fallback and should be treated as a runtime-fairness caveat rather than a
method failure.

## Main round results

### High-level reading

The sixth-round `v1` results support AtlasMTL as a coherent multi-level
annotation framework, but not as a uniform finest-level replacement for the
retained single-level formal benchmark.

The clearest positive result is hierarchy consistency:

- all `16/16` runs had `mean_path_consistency_rate = 1.0`
- all `16/16` runs had `min_path_consistency_rate = 1.0`

The clearest weakness is `PHMap_Lung_Full_v43_light`, which remained the
hardest hierarchy in both finest-level and full-path metrics.

### Overall qualitative ranking

- strongest multi-level case: `mTCA`
- stable deep hierarchy case: `HLCA_Core`
- stable shallow hierarchy case: `DISCO_hPBMCs`
- main hard case: `PHMap_Lung_Full_v43_light`

## Complete per-level results

### `HLCA_Core`

| track    | point                | level       | accuracy | macro_f1 | balanced_accuracy | coverage | covered_accuracy | risk     |
| -------- | -------------------- | ----------- | -------- | -------- | ----------------- | -------- | ---------------- | -------- |
| cpu_core | build_100000_eval10k | ann_level_1 | 0.996000 | 0.993536 | 0.994274          | 1.000000 | 0.996000         | 0.004000 |
| cpu_core | build_100000_eval10k | ann_level_2 | 0.990500 | 0.873269 | 0.865447          | 0.998700 | 0.991789         | 0.008211 |
| cpu_core | build_100000_eval10k | ann_level_3 | 0.961100 | 0.812925 | 0.801819          | 0.997600 | 0.963412         | 0.036588 |
| cpu_core | build_100000_eval10k | ann_level_4 | 0.895500 | 0.781579 | 0.760286          | 0.991300 | 0.903359         | 0.096641 |
| cpu_core | build_100000_eval10k | ann_level_5 | 0.861000 | 0.747120 | 0.726118          | 0.983300 | 0.875623         | 0.124377 |
| cpu_core | predict_100000_10000 | ann_level_1 | 0.996400 | 0.993788 | 0.992878          | 1.000000 | 0.996400         | 0.003600 |
| cpu_core | predict_100000_10000 | ann_level_2 | 0.989700 | 0.861869 | 0.846947          | 0.998100 | 0.991584         | 0.008416 |
| cpu_core | predict_100000_10000 | ann_level_3 | 0.961500 | 0.809480 | 0.793564          | 0.996800 | 0.964587         | 0.035413 |
| cpu_core | predict_100000_10000 | ann_level_4 | 0.897600 | 0.787298 | 0.763253          | 0.989100 | 0.907492         | 0.092508 |
| cpu_core | predict_100000_10000 | ann_level_5 | 0.866700 | 0.744923 | 0.739591          | 0.983400 | 0.881330         | 0.118670 |
| gpu      | build_100000_eval10k | ann_level_1 | 0.996600 | 0.993983 | 0.995611          | 1.000000 | 0.996600         | 0.003400 |
| gpu      | build_100000_eval10k | ann_level_2 | 0.991600 | 0.955370 | 0.935141          | 0.998900 | 0.992692         | 0.007308 |
| gpu      | build_100000_eval10k | ann_level_3 | 0.961900 | 0.813969 | 0.802352          | 0.997700 | 0.964117         | 0.035883 |
| gpu      | build_100000_eval10k | ann_level_4 | 0.896600 | 0.808472 | 0.790287          | 0.991000 | 0.904743         | 0.095257 |
| gpu      | build_100000_eval10k | ann_level_5 | 0.868800 | 0.774340 | 0.764218          | 0.986100 | 0.881047         | 0.118953 |
| gpu      | predict_100000_10000 | ann_level_1 | 0.996200 | 0.993879 | 0.996095          | 1.000000 | 0.996200         | 0.003800 |
| gpu      | predict_100000_10000 | ann_level_2 | 0.990900 | 0.858164 | 0.838341          | 0.998200 | 0.992687         | 0.007313 |
| gpu      | predict_100000_10000 | ann_level_3 | 0.961900 | 0.809356 | 0.784690          | 0.996600 | 0.965182         | 0.034818 |
| gpu      | predict_100000_10000 | ann_level_4 | 0.892100 | 0.780931 | 0.768815          | 0.988900 | 0.902113         | 0.097887 |
| gpu      | predict_100000_10000 | ann_level_5 | 0.860700 | 0.752047 | 0.746899          | 0.980200 | 0.878086         | 0.121914 |

### `PHMap_Lung_Full_v43_light`

| track    | point                | level    | accuracy | macro_f1 | balanced_accuracy | coverage | covered_accuracy | risk     |
| -------- | -------------------- | -------- | -------- | -------- | ----------------- | -------- | ---------------- | -------- |
| cpu_core | build_100000_eval10k | anno_lv1 | 0.995300 | 0.827937 | 0.827860          | 0.999600 | 0.995698         | 0.004302 |
| cpu_core | build_100000_eval10k | anno_lv2 | 0.703700 | 0.752139 | 0.801216          | 0.996400 | 0.706242         | 0.293758 |
| cpu_core | build_100000_eval10k | anno_lv3 | 0.653500 | 0.704075 | 0.726058          | 0.981700 | 0.665682         | 0.334318 |
| cpu_core | build_100000_eval10k | anno_lv4 | 0.576500 | 0.619452 | 0.616008          | 0.793000 | 0.726986         | 0.273014 |
| cpu_core | predict_100000_10000 | anno_lv1 | 0.995600 | 0.994227 | 0.993706          | 0.999900 | 0.995700         | 0.004300 |
| cpu_core | predict_100000_10000 | anno_lv2 | 0.690700 | 0.800067 | 0.845989          | 0.996900 | 0.692848         | 0.307152 |
| cpu_core | predict_100000_10000 | anno_lv3 | 0.634600 | 0.742350 | 0.753755          | 0.986600 | 0.643219         | 0.356781 |
| cpu_core | predict_100000_10000 | anno_lv4 | 0.559900 | 0.616097 | 0.622915          | 0.765800 | 0.731131         | 0.268869 |
| gpu      | build_100000_eval10k | anno_lv1 | 0.995300 | 0.827894 | 0.827989          | 0.999700 | 0.995599         | 0.004401 |
| gpu      | build_100000_eval10k | anno_lv2 | 0.703500 | 0.755054 | 0.800559          | 0.997100 | 0.705546         | 0.294454 |
| gpu      | build_100000_eval10k | anno_lv3 | 0.652800 | 0.705637 | 0.726078          | 0.983100 | 0.664022         | 0.335978 |
| gpu      | build_100000_eval10k | anno_lv4 | 0.577500 | 0.631830 | 0.639773          | 0.815900 | 0.707807         | 0.292193 |
| gpu      | predict_100000_10000 | anno_lv1 | 0.995600 | 0.994217 | 0.993456          | 0.999900 | 0.995700         | 0.004300 |
| gpu      | predict_100000_10000 | anno_lv2 | 0.694100 | 0.788857 | 0.864227          | 0.997100 | 0.696119         | 0.303881 |
| gpu      | predict_100000_10000 | anno_lv3 | 0.631600 | 0.732425 | 0.769417          | 0.984300 | 0.641674         | 0.358326 |
| gpu      | predict_100000_10000 | anno_lv4 | 0.551800 | 0.634550 | 0.656824          | 0.842900 | 0.654645         | 0.345355 |

### `DISCO_hPBMCs`

| track    | point                | level        | accuracy | macro_f1 | balanced_accuracy | coverage | covered_accuracy | risk     |
| -------- | -------------------- | ------------ | -------- | -------- | ----------------- | -------- | ---------------- | -------- |
| cpu_core | build_100000_eval10k | cell_subtype | 0.886000 | 0.782334 | 0.772841          | 0.984600 | 0.899858         | 0.100142 |
| cpu_core | build_100000_eval10k | cell_type    | 0.956000 | 0.799177 | 0.796005          | 0.999300 | 0.956670         | 0.043330 |
| cpu_core | predict_100000_10000 | cell_subtype | 0.883900 | 0.805037 | 0.792931          | 0.983300 | 0.898912         | 0.101088 |
| cpu_core | predict_100000_10000 | cell_type    | 0.953200 | 0.832213 | 0.826061          | 0.999000 | 0.954154         | 0.045846 |
| gpu      | build_100000_eval10k | cell_subtype | 0.886900 | 0.825551 | 0.806850          | 0.986100 | 0.899402         | 0.100598 |
| gpu      | build_100000_eval10k | cell_type    | 0.957600 | 0.866357 | 0.846598          | 0.999100 | 0.958463         | 0.041537 |
| gpu      | predict_100000_10000 | cell_subtype | 0.882700 | 0.805652 | 0.790717          | 0.985100 | 0.896051         | 0.103949 |
| gpu      | predict_100000_10000 | cell_type    | 0.952600 | 0.839162 | 0.826116          | 0.998900 | 0.953649         | 0.046351 |

### `mTCA`

| track    | point                | level            | accuracy | macro_f1 | balanced_accuracy | coverage | covered_accuracy | risk     |
| -------- | -------------------- | ---------------- | -------- | -------- | ----------------- | -------- | ---------------- | -------- |
| cpu_core | build_100000_eval10k | Cell_type_level1 | 0.997200 | 0.982765 | 0.998122          | 1.000000 | 0.997200         | 0.002800 |
| cpu_core | build_100000_eval10k | Cell_type_level2 | 0.963500 | 0.905678 | 0.907875          | 0.997300 | 0.966108         | 0.033892 |
| cpu_core | build_100000_eval10k | Cell_type_level3 | 0.937200 | 0.851276 | 0.845740          | 0.987700 | 0.948871         | 0.051129 |
| cpu_core | predict_100000_10000 | Cell_type_level1 | 0.998000 | 0.993497 | 0.998622          | 1.000000 | 0.998000         | 0.002000 |
| cpu_core | predict_100000_10000 | Cell_type_level2 | 0.968000 | 0.916449 | 0.909894          | 0.997300 | 0.970621         | 0.029379 |
| cpu_core | predict_100000_10000 | Cell_type_level3 | 0.942600 | 0.870098 | 0.860938          | 0.990300 | 0.951833         | 0.048167 |
| gpu      | build_100000_eval10k | Cell_type_level1 | 0.997400 | 0.981580 | 0.998279          | 1.000000 | 0.997400         | 0.002600 |
| gpu      | build_100000_eval10k | Cell_type_level2 | 0.962500 | 0.902093 | 0.905585          | 0.997900 | 0.964526         | 0.035474 |
| gpu      | build_100000_eval10k | Cell_type_level3 | 0.933500 | 0.841100 | 0.847726          | 0.992000 | 0.941028         | 0.058972 |
| gpu      | predict_100000_10000 | Cell_type_level1 | 0.996900 | 0.990240 | 0.997830          | 1.000000 | 0.996900         | 0.003100 |
| gpu      | predict_100000_10000 | Cell_type_level2 | 0.963300 | 0.894930 | 0.899359          | 0.997200 | 0.966005         | 0.033995 |
| gpu      | predict_100000_10000 | Cell_type_level3 | 0.934200 | 0.848143 | 0.846090          | 0.988800 | 0.944782         | 0.055218 |

## Full-path summary

| dataset | track | point | full_path_accuracy | full_path_coverage | full_path_covered_accuracy | mean_path_consistency_rate | min_path_consistency_rate |
| ------- | ----- | ----- | ------------------ | ------------------ | -------------------------- | -------------------------- | ------------------------- |
| `DISCO_hPBMCs` | `cpu_core` | `build_100000_eval10k` | 0.885900 | 0.984500 | 0.899848 | 1.000000 | 1.000000 |
| `DISCO_hPBMCs` | `cpu_core` | `predict_100000_10000` | 0.883900 | 0.982800 | 0.899369 | 1.000000 | 1.000000 |
| `DISCO_hPBMCs` | `gpu` | `build_100000_eval10k` | 0.886800 | 0.986000 | 0.899391 | 1.000000 | 1.000000 |
| `DISCO_hPBMCs` | `gpu` | `predict_100000_10000` | 0.882600 | 0.984700 | 0.896314 | 1.000000 | 1.000000 |
| `HLCA_Core` | `cpu_core` | `build_100000_eval10k` | 0.855500 | 0.973900 | 0.878427 | 1.000000 | 1.000000 |
| `HLCA_Core` | `cpu_core` | `predict_100000_10000` | 0.861000 | 0.972700 | 0.885165 | 1.000000 | 1.000000 |
| `HLCA_Core` | `gpu` | `build_100000_eval10k` | 0.864000 | 0.976900 | 0.884430 | 1.000000 | 1.000000 |
| `HLCA_Core` | `gpu` | `predict_100000_10000` | 0.855400 | 0.970600 | 0.881311 | 1.000000 | 1.000000 |
| `PHMap_Lung_Full_v43_light` | `cpu_core` | `build_100000_eval10k` | 0.573200 | 0.779500 | 0.735343 | 1.000000 | 1.000000 |
| `PHMap_Lung_Full_v43_light` | `cpu_core` | `predict_100000_10000` | 0.555800 | 0.757400 | 0.733826 | 1.000000 | 1.000000 |
| `PHMap_Lung_Full_v43_light` | `gpu` | `build_100000_eval10k` | 0.574200 | 0.804300 | 0.713913 | 1.000000 | 1.000000 |
| `PHMap_Lung_Full_v43_light` | `gpu` | `predict_100000_10000` | 0.549300 | 0.831800 | 0.660375 | 1.000000 | 1.000000 |
| `mTCA` | `cpu_core` | `build_100000_eval10k` | 0.936500 | 0.986200 | 0.949605 | 1.000000 | 1.000000 |
| `mTCA` | `cpu_core` | `predict_100000_10000` | 0.942300 | 0.988800 | 0.952973 | 1.000000 | 1.000000 |
| `mTCA` | `gpu` | `build_100000_eval10k` | 0.932800 | 0.990800 | 0.941461 | 1.000000 | 1.000000 |
| `mTCA` | `gpu` | `predict_100000_10000` | 0.933400 | 0.987100 | 0.945598 | 1.000000 | 1.000000 |

## Finest-level reliability summary

| dataset | track | point | finest_level | coverage | reject_rate | covered_accuracy | risk | ece | brier | aurc | unknown_rate |
| ------- | ----- | ----- | ------------ | -------- | ----------- | ---------------- | ---- | --- | ----- | ---- | ------------ |
| `DISCO_hPBMCs` | `cpu_core` | `build_100000_eval10k` | `cell_subtype` | 0.984600 | 0.015400 | 0.899858 | 0.100142 | 0.019089 | 0.072757 | 0.022526 | 0.015400 |
| `DISCO_hPBMCs` | `cpu_core` | `predict_100000_10000` | `cell_subtype` | 0.983300 | 0.016700 | 0.898912 | 0.101088 | 0.023603 | 0.072405 | 0.022758 | 0.016700 |
| `DISCO_hPBMCs` | `gpu` | `build_100000_eval10k` | `cell_subtype` | 0.986100 | 0.013900 | 0.899402 | 0.100598 | 0.028225 | 0.073604 | 0.023416 | 0.013900 |
| `DISCO_hPBMCs` | `gpu` | `predict_100000_10000` | `cell_subtype` | 0.985100 | 0.014900 | 0.896051 | 0.103949 | 0.033606 | 0.074115 | 0.023012 | 0.014900 |
| `HLCA_Core` | `cpu_core` | `build_100000_eval10k` | `ann_level_5` | 0.983300 | 0.016700 | 0.875623 | 0.124377 | 0.057444 | 0.092052 | 0.029625 | 0.016700 |
| `HLCA_Core` | `cpu_core` | `predict_100000_10000` | `ann_level_5` | 0.983400 | 0.016600 | 0.881330 | 0.118670 | 0.044442 | 0.085056 | 0.026856 | 0.016600 |
| `HLCA_Core` | `gpu` | `build_100000_eval10k` | `ann_level_5` | 0.986100 | 0.013900 | 0.881047 | 0.118953 | 0.055490 | 0.087837 | 0.027336 | 0.013900 |
| `HLCA_Core` | `gpu` | `predict_100000_10000` | `ann_level_5` | 0.980200 | 0.019800 | 0.878086 | 0.121914 | 0.053304 | 0.089573 | 0.029863 | 0.019800 |
| `PHMap_Lung_Full_v43_light` | `cpu_core` | `build_100000_eval10k` | `anno_lv4` | 0.793000 | 0.207000 | 0.726986 | 0.273014 | 0.079181 | 0.152436 | 0.099258 | 0.207000 |
| `PHMap_Lung_Full_v43_light` | `cpu_core` | `predict_100000_10000` | `anno_lv4` | 0.765800 | 0.234200 | 0.731131 | 0.268869 | 0.058376 | 0.139536 | 0.085062 | 0.234200 |
| `PHMap_Lung_Full_v43_light` | `gpu` | `build_100000_eval10k` | `anno_lv4` | 0.815900 | 0.184100 | 0.707807 | 0.292193 | 0.110373 | 0.173803 | 0.118107 | 0.184100 |
| `PHMap_Lung_Full_v43_light` | `gpu` | `predict_100000_10000` | `anno_lv4` | 0.842900 | 0.157100 | 0.654645 | 0.345355 | 0.140620 | 0.182549 | 0.136849 | 0.157100 |
| `mTCA` | `cpu_core` | `build_100000_eval10k` | `Cell_type_level3` | 0.987700 | 0.012300 | 0.948871 | 0.051129 | 0.011397 | 0.036269 | 0.004934 | 0.012300 |
| `mTCA` | `cpu_core` | `predict_100000_10000` | `Cell_type_level3` | 0.990300 | 0.009700 | 0.951833 | 0.048167 | 0.011661 | 0.034985 | 0.004886 | 0.009700 |
| `mTCA` | `gpu` | `build_100000_eval10k` | `Cell_type_level3` | 0.992000 | 0.008000 | 0.941028 | 0.058972 | 0.019944 | 0.042274 | 0.005749 | 0.008000 |
| `mTCA` | `gpu` | `predict_100000_10000` | `Cell_type_level3` | 0.988800 | 0.011200 | 0.944782 | 0.055218 | 0.009463 | 0.038782 | 0.006110 | 0.011200 |

## Main result interpretation

### What worked

- multi-level outputs were usable on all four datasets
- hierarchy consistency was perfect in the executed round:
  - `mean_path_consistency_rate = 1.0`
  - `min_path_consistency_rate = 1.0`
- `mTCA` was the strongest overall case
- `HLCA_Core` remained a credible deep-hierarchy example
- `DISCO_hPBMCs` behaved well as a shallow-hierarchy case

### What did not work well

- the current `v1` configuration did not uniformly beat the retained
  single-level formal AtlasMTL baseline at the finest level
- `PHMap_Lung_Full_v43_light` was clearly the weakest hierarchy
- hierarchy-aware enforcement appears beneficial for consistency, but not
  uniformly beneficial for finest-level headline metrics

### Current manuscript-safe interpretation

The sixth round supports AtlasMTL as a multi-level annotation framework with
strict hierarchy-consistent outputs. It does not yet support claiming that the
current hierarchy-aware multi-level configuration is a direct headline
replacement for the retained single-level formal benchmark rows.

## Sixth-round follow-up probes

### 1. PHMap GPU weight probe

Purpose:

- test whether stronger fine-level weighting improves the weakest hard-case
  dataset without changing the rest of the sixth-round contract

Tested weight:

- `task_weights = [0.2, 0.7, 1.5, 3.0]`

Result versus sixth-round `v1` GPU baseline:

- `build_100000_eval10k`
  - `anno_lv4 macro_f1`: `0.631830 -> 0.633440` (`+0.001610`)
  - `full_path_accuracy`: `0.574200 -> 0.581700` (`+0.007500`)
- `predict_100000_10000`
  - `anno_lv4 macro_f1`: `0.634550 -> 0.653740` (`+0.019190`)
  - `full_path_accuracy`: `0.549300 -> 0.563500` (`+0.014200`)

Interpretation:

- weight optimization clearly matters on `PHMap`
- a fine-level-upweighted configuration is a real positive candidate

### 2. PHMap GPU training-contract probe

Purpose:

- test whether the `PHMap` gain mainly comes from weights or whether
  `100 epochs` / `lr=1e-3` are also needed

Fixed:

- `task_weights = [0.2, 0.7, 1.5, 3.0]`

Scanned:

- `50 epochs, 3e-4`
- `100 epochs, 3e-4`
- `50 epochs, 1e-3`
- `100 epochs, 1e-3`

Outcome:

- the strongest and most stable gain came from the weight change itself
- neither `100 epochs` nor `lr=1e-3` provided a stable enough additional win
- the best practical candidate remained:
  - `task_weights = [0.2, 0.7, 1.5, 3.0]`
  - `num_epochs = 50`
  - `learning_rate = 3e-4`

### 3. HLCA GPU weight probe

Purpose:

- test whether fine-level upweight is only a `PHMap` effect or can generalize
  to a deeper 5-level hierarchy

Tested:

- `uniform = [1, 1, 1, 1, 1]`
- `hlca_lv5strong_a = [0.2, 0.5, 1.0, 1.8, 3.0]`
- `hlca_lv5strong_b = [0.3, 0.7, 1.2, 2.0, 3.0]`

Key finest-level `ann_level_5 macro_f1` results:

- original sixth-round GPU baseline
  - `build`: `0.774340`
  - `predict`: `0.752047`
- `uniform` probe
  - `build`: `0.772677`
  - `predict`: `0.739650`
- `hlca_lv5strong_a`
  - `build`: `0.746465`
  - `predict`: `0.788413`
- `hlca_lv5strong_b`
  - `build`: `0.763807`
  - `predict`: `0.744270`

Interpretation:

- fine-level upweight is not only a `PHMap` effect
- but the best weight pattern is not obviously portable across datasets or
  across build/predict points
- a single global weight schedule for all sixth-round datasets is therefore not
  yet justified

### 4. Weighted v2 global redesign attempt

We attempted to extend the weighted redesign across all sixth-round datasets,
but the first practical blocker was structural:

- the confirmed `PHMap` weight vector `[0.2, 0.7, 1.5, 3.0]` is only valid for
  4-level datasets
- `HLCA_Core` has 5 levels
- `mTCA` has 3 levels
- `DISCO_hPBMCs` has 2 levels

So a direct one-vector-for-all-datasets redesign was not rigorous enough to
continue.

## Problems and open questions

### Problem 1: finest-level performance is not uniformly improved

The sixth-round `v1` multi-level setting provides extra hierarchy structure but
does not uniformly outperform the retained single-level formal benchmark on the
finest label.

### Problem 2: `PHMap` is the main hard case

`PHMap_Lung_Full_v43_light` shows:

- the weakest finest-level performance
- the lowest full-path performance
- the highest reject/unknown pressure

This dataset is the most important benchmark for any redesign.

### Problem 3: hierarchy enforcement is a tradeoff, not a free gain

The current implementation uses `enforce_hierarchy=True` as a post-prediction
consistency pass. This is likely helping path consistency while sacrificing some
finest-level coverage and headline metrics.

### Problem 4: task weights are high-impact and dataset-sensitive

The follow-up probes strongly suggest:

- `uniform` weighting is probably suboptimal
- fine-level upweight can help substantially
- but the best weight schedule may differ by dataset depth and task difficulty

## Terms and definitions

### `full_path_accuracy`

The fraction of cells whose predicted labels are correct at **all hierarchy
levels simultaneously**. For a 4-level hierarchy, a cell counts as correct only
if `lv1`, `lv2`, `lv3`, and `lv4` are all correct.

### `full_path_coverage`

The fraction of cells for which a full hierarchy path remains available after
rejection / `Unknown` handling.

### `full_path_covered_accuracy`

Accuracy computed only on the subset of cells that still have a valid
non-rejected full hierarchy path.

### `coverage`

The fraction of cells that receive a non-rejected prediction at a given level.

### `reject_rate`

The fraction of cells rejected at a given level. In this round it is effectively
the complement of `coverage`.

### `covered_accuracy`

Accuracy computed only on the cells that were not rejected.

### `risk`

The error rate on the covered subset. In these outputs it is effectively
`1 - covered_accuracy`.

### `unknown_rate`

The fraction of cells assigned `Unknown` at a given level.

### `mean_path_consistency_rate`

The mean parent-child consistency across hierarchy edges. A value of `1.0`
means no observed parent-child contradiction remained after the hierarchy pass.

### `min_path_consistency_rate`

The worst edge-level consistency rate among all hierarchy edges in a run.

### `hierarchy-aware` / `enforce_hierarchy=True`

A post-prediction consistency policy. It does not retrain the model. It checks
whether child predictions are compatible with predicted parent labels and can
force incompatible child predictions to `Unknown`.

### `runtime_fairness_degraded`

A flag indicating that the run was completed under a restricted execution mode
that weakens runtime fairness interpretation. In the sixth round this applied to
all CPU runs because `joblib` fell back to serial mode.

## Current recommendation before expert feedback

The expert discussion should focus on three questions:

1. Should the manuscript position the multi-level round as a distinct
   capability benchmark rather than a direct finest-level replacement benchmark?
2. Should the next redesign prioritize `task_weights` before any new
   feature-space or optimizer change?
3. Should `hierarchy on/off` predict-only ablation be treated as the next
   cheapest clarification step for the current sixth-round models?
