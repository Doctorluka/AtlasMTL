# AtlasMTL Ablation Summary

Run scope:

- dataset: sampled real benchmark bundle
  - reference: `data/test_adatas/sampled_adata_10k.h5ad`
  - query: `data/test_adatas/sampled_adata_3000.h5ad`
- labels: `anno_lv1`, `anno_lv2`, `anno_lv3`, `anno_lv4`
- shared evaluation focus: `anno_lv4`
- hierarchy enforcement: on
- KNN correction: off
- total variants: `24`
  - device: `cpu`, `cuda`
  - feature space: `whole`, `hvg3000`, `hvg6000`
  - input transform: `binary`, `float`
  - task weights: `uniform`, `phmap`

Top-performing variants by `anno_lv4` accuracy:

1. `atlasmtl_cuda_hvg6000_binary_phmap`
   - accuracy: `0.7730`
   - macro-F1: `0.6720`
   - full-path accuracy: `0.7617`
2. `atlasmtl_cpu_hvg6000_binary_phmap`
   - accuracy: `0.7677`
   - macro-F1: `0.6618`
   - full-path accuracy: `0.7600`
3. `atlasmtl_cpu_whole_binary_phmap`
   - accuracy: `0.7663`
   - macro-F1: `0.6673`
   - full-path accuracy: `0.7603`
4. `atlasmtl_cuda_whole_binary_phmap`
   - accuracy: `0.7657`
   - macro-F1: `0.6601`
   - full-path accuracy: `0.7550`

Stable observations:

- all `24` runs completed successfully
- all runs kept edge-level hierarchy consistency at `1.0`
- `binary` was consistently stronger than `float`
- `phmap` task weights were consistently stronger than `uniform`
- `hvg6000` gave the best observed tradeoff and the best top-line result

Average trends across the full grid:

- by input transform:
  - `binary`: average `anno_lv4` accuracy `0.7530`
  - `float`: average `anno_lv4` accuracy `0.6639`
- by task weights:
  - `phmap`: average `anno_lv4` accuracy `0.7205`
  - `uniform`: average `anno_lv4` accuracy `0.6964`
- by device:
  - `cpu`: average `anno_lv4` accuracy `0.7081`
  - `cuda`: average `anno_lv4` accuracy `0.7087`

Resource summary:

- average train time:
  - `cpu`: `10.7532 s`
  - `cuda`: `3.4301 s`
- average predict time:
  - `cpu`: `0.1189 s`
  - `cuda`: `0.0774 s`
- representative peak RSS:
  - `whole`: about `4.84` to `5.06 GB`
  - `hvg6000`: about `3.42 GB`
- representative peak GPU memory:
  - `whole cuda`: about `0.1469 GB`
  - `hvg6000 cuda`: about `0.0526 GB`

Primary output files:

- `~/tmp/atlasmtl_real_mapping_benchmark_20260301/ablation_optimization/metrics.json`
- `~/tmp/atlasmtl_real_mapping_benchmark_20260301/ablation_optimization/benchmark_report.md`
- `~/tmp/atlasmtl_real_mapping_benchmark_20260301/ablation_optimization/paper_tables/atlasmtl_ablation_accuracy.csv`
- `~/tmp/atlasmtl_real_mapping_benchmark_20260301/ablation_optimization/paper_tables/atlasmtl_ablation_resources.csv`
- `~/tmp/atlasmtl_real_mapping_benchmark_20260301/ablation_optimization/paper_tables/atlasmtl_ablation_tradeoff.csv`
