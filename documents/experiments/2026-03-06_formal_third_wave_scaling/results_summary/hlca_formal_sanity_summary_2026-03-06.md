# HLCA formal sanity summary

Date: `2026-03-06`

This summary records the first formal execution sanity check for `HLCA_Core`
under the third-wave formal scaling round.

## Scope

Sanity points:

- build scaling: `build=100k`, `eval=10k`
- predict scaling: `fixed_build=100k`, `predict=10k`

Device groups:

- CPU group: `atlasmtl`, `reference_knn`, `celltypist`, `singler`, `symphony`,
  `seurat_anchor_transfer`
- GPU group: `atlasmtl`, `scanvi`

## Execution outcome

GPU:

- `build100k -> eval10k`: completed
- `predict100k -> 10k`: completed

CPU:

- `build100k -> eval10k`: partially completed
- `predict100k -> 10k`: not started
- `seurat_anchor_transfer` was manually stopped after exceeding the new
  `60-minute` runtime guardrail

## Completed result table

| Track | Device | Method | Backend | Accuracy | Macro-F1 | Train s | Predict s | Train peak RSS GB | Predict peak RSS GB | Train peak GPU GB | Predict peak GPU GB | Status |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| build100k->eval10k | CPU | `atlasmtl` | `atlasmtl_native` | 0.8720 | 0.7791 | 37.0161 | 0.0959 | 5.1089 | 6.0969 | 0.0000 | 0.0000 | complete |
| build100k->eval10k | CPU | `reference_knn` | `reference_knn_native` | 0.7992 | 0.5948 | 1.2555 | 19.0054 | 3.7588 | 3.7588 | 0.0000 | 0.0000 | complete |
| build100k->eval10k | CPU | `celltypist` | `formal_native` | 0.8681 | 0.7905 | 0.0015 | 0.3858 | 1.0151 | 1.3221 | 0.0000 | 0.0000 | complete |
| build100k->eval10k | CPU | `singler` | `singler_native` | 0.7983 | 0.7474 | 1616.2645 | 1616.2645 | 3.6964 | 3.6964 | 0.0000 | 0.0000 | complete |
| build100k->eval10k | CPU | `symphony` | `symphony_native` | 0.7086 | 0.4910 | 65.6406 | 65.6406 | 6.7812 | 6.7812 | 0.0000 | 0.0000 | complete |
| build100k->eval10k | GPU | `atlasmtl` | `atlasmtl_native` | 0.8669 | 0.7555 | 7.9982 | 0.0561 | 5.5648 | 6.8356 | 0.0849 | 0.0444 | complete |
| build100k->eval10k | GPU | `scanvi` | `scanvi_native` | 0.8877 | 0.8075 | 300.0190 | 14.5083 | 3.6674 | 3.6696 | 0.1833 | 0.1536 | complete |
| predict100k->10k | GPU | `atlasmtl` | `atlasmtl_native` | 0.8653 | 0.7567 | 8.2232 | 0.0544 | 5.5771 | 6.8652 | 0.0855 | 0.0444 | complete |
| predict100k->10k | GPU | `scanvi` | `scanvi_native` | 0.8905 | 0.8024 | 299.7297 | 14.4880 | 3.6589 | 3.6611 | 0.1808 | 0.1536 | complete |

## Incomplete item

| Track | Device | Method | Status | Note |
| --- | --- | --- | --- | --- |
| build100k->eval10k | CPU | `seurat_anchor_transfer` | `manual_stop_long_runtime` | manually stopped after about `1h15m`; no final `summary.csv` emitted |

## Key interpretation

- formal manifests are valid for both CPU and GPU tracks
- GPU sanity is fully passed for the two `100k` sanity points
- CPU sanity is usable as a partial comparator/resource snapshot, but not as a
  complete all-method formal summary
- `seurat_anchor_transfer` is now a known long-runtime CPU comparator and must
  be scheduled with the `60-minute` stop rule in mind

## Primary result roots

- CPU partial:
  - `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/HLCA_Core/benchmark/sanity_check/cpu/build100k_eval10k/`
- GPU build:
  - `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/HLCA_Core/benchmark/sanity_check/gpu/build100k_eval10k/`
- GPU predict:
  - `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/HLCA_Core/benchmark/sanity_check/gpu/predict100k_10k/`

## Related records

- `results_summary/execution_report_2026-03-06_formal_hlca_sanity_start.md`
- `documents/experiments/2026-03-06_formal_third_wave_round_status.md`
