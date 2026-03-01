# Multi-Level AtlasMTL Summary

Runtime artifact directory:

- `~/tmp/atlasmtl_real_mapping_benchmark_20260301/multilevel_atlasmtl/`

Primary artifacts:

- model:
  `~/tmp/atlasmtl_real_mapping_benchmark_20260301/multilevel_atlasmtl/atlasmtl_multilevel_model.pth`
- annotated query:
  `~/tmp/atlasmtl_real_mapping_benchmark_20260301/multilevel_atlasmtl/query_with_predictions.h5ad`
- prediction table:
  `~/tmp/atlasmtl_real_mapping_benchmark_20260301/multilevel_atlasmtl/predictions_full.csv`
- metrics:
  `~/tmp/atlasmtl_real_mapping_benchmark_20260301/multilevel_atlasmtl/metrics.json`

Per-level metrics:

| level | accuracy | macro_f1 | balanced_accuracy |
| --- | ---: | ---: | ---: |
| anno_lv1 | 0.9877 | 0.9855 | 0.9858 |
| anno_lv2 | 0.9353 | 0.8569 | 0.8822 |
| anno_lv3 | 0.8700 | 0.7473 | 0.7477 |
| anno_lv4 | 0.7437 | 0.6274 | 0.5990 |

Hierarchy metrics:

- edge consistency:
  - `anno_lv2 -> anno_lv1`: `1.0000`
  - `anno_lv3 -> anno_lv2`: `1.0000`
  - `anno_lv4 -> anno_lv3`: `1.0000`
- full-path accuracy: `0.7330`
- full-path coverage: `0.8993`
- covered full-path accuracy: `0.8150`

Interpretation:

- the expected difficulty trend appears clearly:
  coarse labels are nearly saturated, while performance decreases as the task
  moves toward `anno_lv4`
- hierarchy enforcement is functioning as intended on this run
- this sampled real-data run supports the project claim that AtlasMTL is more
  than four independent flat classifiers because path-consistency is explicitly
  enforced and auditable

Explicit exclusion:

- `knn_correction = off`
- reason:
  the sampled datasets do not provide usable coordinate targets in `obsm`
