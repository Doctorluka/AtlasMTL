# Formal Resource Table

| group | method | device_used | method_backend_path | runtime_fairness_degraded | train_elapsed_seconds | predict_elapsed_seconds | train_process_peak_rss_gb | predict_process_peak_rss_gb | train_gpu_peak_memory_gb | predict_gpu_peak_memory_gb | train_items_per_second | predict_items_per_second | effective_threads_observed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | atlasmtl | cpu | atlasmtl_native | True | 2.1903 | 0.054 | 1.4604 | 1.4978 | 0.0 | 0.0 | 32872.4158 | 92540.8534 | 10.0 |
| cpu | celltypist | cpu | formal_native | True | 0.0018 | 0.2488 | 0.9391 | 1.0881 | 0.0 | 0.0 | 549.8541 | 20099.338 | nan |
| cpu | reference_knn | cpu | reference_knn_native | True | 0.1328 | 1.2315 | 1.1695 | 1.1827 | 0.0 | 0.0 | 75279.8257 | 4060.0574 | nan |
| cpu | seurat_anchor_transfer | cpu | TransferData-only | True | 85.442 | 85.442 | 2.8141 | 2.8141 | 0.0 | 0.0 | 58.5193 | 58.5193 | nan |
| cpu | singler | cpu | singler_native | True | 92.8093 | 92.8093 | 1.1018 | 1.1018 | 0.0 | 0.0 | 53.8739 | 53.8739 | nan |
| cpu | symphony | cpu | symphony_native | True | 18.8621 | 18.8621 | 1.5549 | 1.5549 | 0.0 | 0.0 | 265.0813 | 265.0813 | nan |
| gpu | atlasmtl | cuda | atlasmtl_native | False | 1.4069 | 0.0288 | 1.8564 | 1.9348 | 0.0351 | 0.0229 | 51177.4448 | 173700.3466 | 10.0 |
| gpu | scanvi | cuda | scanvi_native | False | 36.5337 | 7.0884 | 2.1051 | 2.1051 | 0.1072 | 0.101 | 273.7201 | 705.3766 | nan |
