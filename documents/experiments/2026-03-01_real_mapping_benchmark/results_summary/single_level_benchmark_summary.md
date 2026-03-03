# Single-Level Benchmark Summary

Shared target label:

- `anno_lv4`

Completed benchmark outputs:

- final all-method benchmark bundle:
  `~/tmp/atlasmtl_real_mapping_benchmark_20260301/single_level_benchmark/all_methods_final_v2/`
- paper tables:
  `~/tmp/atlasmtl_real_mapping_benchmark_20260301/single_level_benchmark/all_methods_final_v2/paper_tables/`
- markdown report:
  `~/tmp/atlasmtl_real_mapping_benchmark_20260301/single_level_benchmark/all_methods_final_v2/benchmark_report.md`

Completed methods and current headline metrics:

| method | accuracy | macro_f1 | balanced_accuracy | note |
| --- | ---: | ---: | ---: | --- |
| atlasmtl | 0.7467 | 0.6381 | 0.6068 | completed through benchmark runner |
| reference_knn | 0.3630 | 0.2917 | 0.2606 | completed through benchmark runner |
| celltypist | 0.7900 | 0.7093 | 0.7104 | completed through benchmark runner after dense-query compatibility fix |
| scanvi | 0.5773 | 0.3161 | 0.3409 | completed through benchmark runner |
| singler | 0.6850 | 0.5990 | 0.6360 | completed through benchmark runner |
| symphony | 0.6160 | 0.4838 | 0.4875 | completed through benchmark runner |
| seurat_anchor_transfer | 0.7343 | 0.5888 | 0.5854 | completed through benchmark runner on the predecessor Seurat anchor-transfer path |

Current interpretation:

- `celltypist` is the strongest completed method on this sampled real-data
  `anno_lv4` task under the current prepared-expression input contract.
- `atlasmtl` is the strongest method trained directly inside the shared
  benchmark runner among the non-external-learning baselines.
- `scanvi` improves over the local `reference_knn` baseline but remains behind
  `atlasmtl`, `singler`, and `celltypist`.
- `singler` remains a competitive single-level baseline.
- this comparator result corresponds to the Seurat anchor-transfer path and
  should now be reported under `seurat_anchor_transfer`.

Important scope note:

- this benchmark is on sampled reference/query data, not the final formal paper
  benchmark
- KNN correction is disabled in this single-level run
