# HLCA second-wave scale-out benchmark record (`10k` heldout)

- date: `2026-03-05`
- stage: `benchmark_recovery`
- dataset: `HLCA_Core`
- command (single-method recovery run):
  - `benchmark/pipelines/run_benchmark.py --dataset-manifest /home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/benchmark/group_split_v2_train100k_test10k/all_methods_v1/runtime_manifest.yaml --output-dir /home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/benchmark/group_split_v2_train100k_test10k/all_methods_v1/runs/seurat_anchor_transfer --methods seurat_anchor_transfer --device cpu`
- prepared input root:
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/prepared/group_split_v2_train100k_test10k_nested5k/`
- benchmark output root:
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/benchmark/group_split_v2_train100k_test10k/all_methods_v1/`

## Interruption and fix

- observed issue:
  - the original all-method wrapper run was interrupted unexpectedly on the
    server before final benchmark status materialization
  - top-level `scaleout_status.json` was missing even though six methods had
    completed and `seurat_anchor_transfer` had started
- corrective action:
  - reran `seurat_anchor_transfer` as a single method on the same runtime
    manifest and output root
  - verified method outputs:
    - `runs/seurat_anchor_transfer/metrics.json`
    - `runs/seurat_anchor_transfer/summary.csv`
    - `runs/seurat_anchor_transfer/seurat_anchor_transfer/metadata.json`
  - manually reconstructed top-level status file:
    - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/benchmark/group_split_v2_train100k_test10k/all_methods_v1/scaleout_status.json`

## Final status

- `7/7` methods marked as `success` in reconstructed `scaleout_status.json`
- final comparator set present:
  - `atlasmtl`
  - `reference_knn`
  - `celltypist`
  - `scanvi`
  - `singler`
  - `symphony`
  - `seurat_anchor_transfer`

## Key file paths for audit

- reconstructed status:
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/benchmark/group_split_v2_train100k_test10k/all_methods_v1/scaleout_status.json`
- method summaries:
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/benchmark/group_split_v2_train100k_test10k/all_methods_v1/runs/atlasmtl/summary.csv`
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/benchmark/group_split_v2_train100k_test10k/all_methods_v1/runs/reference_knn/summary.csv`
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/benchmark/group_split_v2_train100k_test10k/all_methods_v1/runs/celltypist/summary.csv`
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/benchmark/group_split_v2_train100k_test10k/all_methods_v1/runs/scanvi/summary.csv`
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/benchmark/group_split_v2_train100k_test10k/all_methods_v1/runs/singler/summary.csv`
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/benchmark/group_split_v2_train100k_test10k/all_methods_v1/runs/symphony/summary.csv`
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/benchmark/group_split_v2_train100k_test10k/all_methods_v1/runs/seurat_anchor_transfer/summary.csv`

## Next action

- sync this completion state back into:
  - `documents/experiments/2026-03-04_second_wave_round_status.md`
  - `documents/experiments/2026-03-04_second_wave_execution_template.md`
