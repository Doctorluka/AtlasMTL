# Vento second-wave scale-out benchmark record (`5k` heldout)

- date: `2026-03-04`
- stage: `benchmark`
- dataset: `Vento`
- command:
  - `documents/experiments/common/run_reference_heldout_scaleout_benchmark.py --dataset-manifest documents/experiments/2026-03-04_projectsvr_vento_benchmark/manifests/reference_heldout/Vento__annotation__scaleout_runtime_5k_v1.yaml --output-dir /home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/Vento/benchmark/group_split_v2_train50k_test5k_nested/all_methods_v1 --methods atlasmtl reference_knn celltypist scanvi singler symphony seurat_anchor_transfer --device cpu`
- prepared input root:
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/Vento/prepared/group_split_v2_train50k_test10k_nested5k/`
- benchmark output root:
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/Vento/benchmark/group_split_v2_train50k_test5k_nested/all_methods_v1/`
- status summary:
  - all `7` methods completed successfully
- important runtime files:
  - `scaleout_status.json`
  - `runtime_manifest.yaml`
  - `runtime_manifest_celltypist.yaml`
  - `runs/<method>/summary.csv`
  - `runs/<method>/metrics.json`
- main engineering issue encountered:
  - no new blocking issue beyond the expected long runtimes of `scanvi`,
    `singler`, and `seurat_anchor_transfer`
- resolution:
  - reused the same shared benchmark wrapper and the already materialized
    `50k/10k/5k` prepared assets from the `Vento` preparation stage
- remaining caveat:
  - `seurat_anchor_transfer` completed successfully but remained substantially
    weaker than the leading methods on the nested `5k` query
- next action:
  - use the `10k` and `5k` benchmark records together when drafting the final
    Vento round summary and paper tables
