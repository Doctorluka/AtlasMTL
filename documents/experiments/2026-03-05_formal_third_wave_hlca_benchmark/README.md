# Formal third-wave HLCA benchmark dossier

Date: `2026-03-05`

This dossier tracks the formal third-wave pilot run for `HLCA_Core` using:

- `train10k` reference model build
- `test5k` heldout prediction
- split execution into two explicit groups:
  - CPU group: all methods except `scanvi`
  - GPU group: `atlasmtl` and `scanvi`

Status note:

- the `2026-03-05` run is a historical formal pilot executed before the final
  `scanvi` (`2026-03-06`) and `atlasmtl` (`2026-03-07`) parameter locks were
  completed.
- the original pilot manifests are preserved as `*_v1.yaml`.
- rerun-ready manifests aligned to the locked formal defaults are provided as
  `*_v2.yaml`.

## Structure

- `manifests/reference_heldout/`
  - CPU and GPU runtime manifests for the formal `train10k/test5k` run
- `scripts/`
  - split materialization script
  - environment snapshot script
  - CPU/GPU group launch scripts
- `results_summary/`
  - execution report
  - experiment record
  - error/fix record
  - environment version artifacts

## Runtime roots

- prepared:
  - `/tmp/atlasmtl_benchmarks/2026-03-05/reference_heldout/HLCA_Core/prepared/formal_train10k_test5k/`
- CPU benchmark:
  - `/tmp/atlasmtl_benchmarks/2026-03-05/reference_heldout/HLCA_Core/benchmark/formal_train10k_test5k/cpu_group_v1/`
- GPU benchmark:
  - `/tmp/atlasmtl_benchmarks/2026-03-05/reference_heldout/HLCA_Core/benchmark/formal_train10k_test5k/gpu_group_v1/`
