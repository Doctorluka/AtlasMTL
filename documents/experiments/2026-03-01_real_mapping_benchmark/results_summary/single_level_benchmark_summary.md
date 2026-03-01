# Single-Level Benchmark Summary

Shared target label:

- `anno_lv4`

Completed benchmark outputs:

- Python-method benchmark bundle:
  `~/tmp/atlasmtl_real_mapping_benchmark_20260301/single_level_benchmark/python_methods_v2/`
- paper tables:
  `~/tmp/atlasmtl_real_mapping_benchmark_20260301/single_level_benchmark/python_methods_v2/paper_tables/`
- markdown report:
  `~/tmp/atlasmtl_real_mapping_benchmark_20260301/single_level_benchmark/python_methods_v2/benchmark_report.md`

Completed methods and current headline metrics:

| method | accuracy | macro_f1 | balanced_accuracy | note |
| --- | ---: | ---: | ---: | --- |
| atlasmtl | 0.7483 | 0.6250 | 0.5998 | completed through benchmark runner |
| reference_knn | 0.3630 | 0.2917 | 0.2606 | completed through benchmark runner |
| scanvi | 0.6003 | 0.3427 | 0.3660 | completed through benchmark runner |
| singler | 0.6850 | 0.5990 | 0.6360 | predictions completed; metrics computed from exported predictions |
| symphony | 0.6160 | 0.4838 | 0.4875 | predictions completed; metrics computed from exported predictions |

Current interpretation:

- `atlasmtl` is the strongest completed method on the sampled real-data
  `anno_lv4` task.
- `scanvi` improves substantially over the local `reference_knn` baseline but
  remains behind `atlasmtl`.
- `singler` performs competitively on accuracy and balanced accuracy among the
  completed non-atlas baselines.
- `symphony` is below `atlasmtl` and `singler` on this sampled setup.

Methods still unresolved in this run:

- `azimuth`
  - the combined R-method benchmark invocation has not completed yet
  - partial files exist under
    `~/tmp/atlasmtl_real_mapping_benchmark_20260301/single_level_benchmark/r_methods/azimuth/`
- `celltypist`
  - local model training is not closed yet for this run
  - the real-data model output path is reserved at
    `~/tmp/atlasmtl_real_mapping_benchmark_20260301/prepared/celltypist_anno_lv4.pkl`

Important scope note:

- this benchmark is on sampled reference/query data, not the final formal paper
  benchmark
- KNN correction is disabled in this single-level run
