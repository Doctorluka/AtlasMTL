# Execution report: HLCA formal sanity start

Date: `2026-03-06`

This report records the start of the first formal execution sanity check for
the third-wave scaling round.

## Scope

Dataset:

- `HLCA_Core`

Sanity points:

- build scaling: `build=100k`, `eval=10k`
- predict scaling: `fixed_build=100k`, `predict=10k`

Device groups:

- CPU group:
  - `atlasmtl`
  - `reference_knn`
  - `celltypist`
  - `singler`
  - `symphony`
  - `seurat_anchor_transfer`
- GPU group:
  - `atlasmtl`
  - `scanvi`

## Execution scripts

- `documents/experiments/2026-03-06_formal_third_wave_scaling/scripts/run_formal_hlca_cpu_sanity.sh`
- `documents/experiments/2026-03-06_formal_third_wave_scaling/scripts/run_formal_hlca_gpu_sanity.sh`

## Manifest paths

CPU:

- `documents/experiments/2026-03-06_formal_third_wave_scaling/manifests/reference_heldout/HLCA_Core__ann_level_5__formal_build_scaling_cpu_build100000_eval10k_v1.yaml`
- `documents/experiments/2026-03-06_formal_third_wave_scaling/manifests/reference_heldout/HLCA_Core__ann_level_5__formal_predict_scaling_cpu_build100000_predict10000_v1.yaml`

GPU:

- `documents/experiments/2026-03-06_formal_third_wave_scaling/manifests/reference_heldout/HLCA_Core__ann_level_5__formal_build_scaling_gpu_build100000_eval10k_v1.yaml`
- `documents/experiments/2026-03-06_formal_third_wave_scaling/manifests/reference_heldout/HLCA_Core__ann_level_5__formal_predict_scaling_gpu_build100000_predict10000_v1.yaml`

## Output roots

- CPU: `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/HLCA_Core/benchmark/sanity_check/cpu/`
- GPU: `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/HLCA_Core/benchmark/sanity_check/gpu/`

## Goal

This sanity check validates:

- formal manifest correctness
- full execution wrapper correctness
- CellTypist formal training path inside the new formal manifests
- fairness metadata landing
- CPU and GPU group separation

## Update: partial stop after long CPU runtime

Update time: `2026-03-06`

Observed execution status:

- GPU group completed both sanity points successfully:
  - `build100k -> eval10k`
  - `predict100k -> 10k`
- CPU group completed partial results for:
  - `atlasmtl`
  - `reference_knn`
  - `celltypist`
  - `singler`
  - `symphony`
- CPU group did not complete `seurat_anchor_transfer` at the first sanity point.

Stop decision:

- The CPU sanity run was manually stopped after `seurat_anchor_transfer` remained
  active for about `1h15m` on `HLCA 100k -> 10k`.
- The stop was applied to the running wrapper chain and no CPU predict-scaling
  sanity point was started afterward.

Interpretation:

- This is recorded as a long-runtime partial stop, not as a comparator crash.
- The current CPU sanity run should be treated as partially complete and
  unsuitable for a full CPU-group headline summary.

Preserved result roots:

- CPU partial root:
  - `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/HLCA_Core/benchmark/sanity_check/cpu/build100k_eval10k/`
- GPU completed roots:
  - `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/HLCA_Core/benchmark/sanity_check/gpu/build100k_eval10k/`
  - `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/HLCA_Core/benchmark/sanity_check/gpu/predict100k_10k/`

Next handling rule:

- Keep the completed CPU partial artifacts for inspection.
- Treat `seurat_anchor_transfer` as a special long-runtime CPU comparator when
  planning the full formal execution order.
