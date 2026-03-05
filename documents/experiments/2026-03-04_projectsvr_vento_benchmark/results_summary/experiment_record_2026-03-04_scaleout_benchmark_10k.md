# Vento second-wave scale-out benchmark record (`10k` heldout)

- date: `2026-03-04`
- stage: `benchmark`
- dataset: `Vento`
- command:
  - `documents/experiments/common/run_reference_heldout_scaleout_benchmark.py --dataset-manifest documents/experiments/2026-03-04_projectsvr_vento_benchmark/manifests/reference_heldout/Vento__annotation__scaleout_runtime_10k_v1.yaml --output-dir /home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/Vento/benchmark/group_split_v2_train50k_test10k/all_methods_v1 --methods atlasmtl reference_knn celltypist scanvi singler symphony seurat_anchor_transfer --device cpu`
- prepared input root:
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/Vento/prepared/group_split_v2_train50k_test10k_nested5k/`
- benchmark output root:
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/Vento/benchmark/group_split_v2_train50k_test10k/all_methods_v1/`
- status summary:
  - all `7` methods completed successfully
- important runtime files:
  - `scaleout_status.json`
  - `runtime_manifest.yaml`
  - `runtime_manifest_celltypist.yaml`
  - `runs/<method>/summary.csv`
  - `runs/<method>/metrics.json`
- main engineering issue encountered:
  - previous background prep attempts for `Vento` had left only logs and pid
    files, so this round was rerun in the foreground to re-materialize the
    prepared assets before benchmark execution
- resolution:
  - reran `documents/experiments/2026-03-04_projectsvr_vento_benchmark/scripts/run_prepare_scaleout.sh`
    in the foreground
  - confirmed `50k/10k/5k` prepared outputs and then launched the shared
    benchmark wrapper on the `10k` runtime manifest
- remaining caveat:
  - `seurat_anchor_transfer` completed successfully but remained on the
    `seurat_anchor_transfer_transferdata` backend and underperformed strongly
    relative to the leading methods
- next action:
  - run the nested `5k` benchmark and then update the round-level status files
