# Third-wave fairness protocol

Date: `2026-03-05`

This protocol defines stricter fairness controls for the third-wave benchmark
round. It builds on second-wave engineering completion and targets
paper-facing runtime comparability.

## Goal

Primary goal for this round:

- keep label-quality comparison valid
- make runtime/resource comparison auditable and fair

This protocol does not replace dataset-level biological constraints. It adds a
resource-control layer on top of existing benchmark contracts.

## Fairness dimensions

The round must explicitly control these dimensions:

1. execution device policy (`cpu` / `cuda`)
2. effective thread policy (`n_jobs`, BLAS threads, R-side threads)
3. input matrix contract per method
4. train/predict sample-size contract
5. fallback-path labeling

## Device policy

Use one of the two allowed policies and declare it in each run record:

- `cpu_only_strict`: all methods on CPU
- `mixed_backend_labeled`: methods use their native backend capability, but
  every method must record actual `device_used`

For the formal third-wave scaling round, use:

- `mixed_backend_labeled` for the combined headline round
- explicit CPU-only sub-analyses only when the selected method set supports it

Additional locked rules for this round:

- `scanvi` remains GPU-only
- `atlasmtl_cpu` and `atlasmtl_gpu` must remain separate runtime/resource rows

## Thread policy

Before every formal run, fix thread environment variables in the launch shell:

- `OMP_NUM_THREADS`
- `MKL_NUM_THREADS`
- `OPENBLAS_NUM_THREADS`
- `NUMEXPR_NUM_THREADS`

Recommended first strict setting: all fixed to `8`.

Also record method-level parallel settings:

- Python comparators (`n_jobs`, `num_threads`)
- R comparators (BLAS/OpenMP and package-level thread controls when available)

## Known environment caveat

In restricted/sandboxed execution, `joblib` may emit:

- `joblib will operate in serial mode (Errno 13: Permission denied)`

When this appears:

- mark the run as `runtime_fairness_degraded=true`
- keep it for engineering functionality checks only
- do not use it as final paper runtime evidence

## Sample-size contract

For the formal third-wave scaling round:

- main-panel datasets: `HLCA_Core`, `PHMap_Lung_Full_v43_light`, `mTCA`,
  `DISCO_hPBMCs`
- supplementary dataset: `Vento`
- excluded: `cd4`, `cd8`

Build scaling:

- build grid: `10k / 20k / 30k / 50k / 100k / 150k / 200k / 300k`
- fixed query: one dedicated `build_eval_fixed_10k`

Predict scaling:

- fixed build: reuse the `100k` build artifact from build scaling
- predict grid: `1k / 3k / 5k / 8k / 10k / 15k / 20k`
- optional tail: `50k`

Locked separation rule:

- build-scaling fixed `10k` and predict-scaling `10k` must be different
  heldout subsets

## Fallback labeling rule

Every comparator must expose whether the native path or fallback path was used.

Current mandatory labels:

- `celltypist`: `formal_native` / `wrapped_logreg` / `formal_with_compat_shim`
- `seurat_anchor_transfer`: `MapQuery` / `TransferData-only`

Fallback results are valid engineering outputs but must not be mixed into
headline runtime tables without explicit labeling.

## Required reporting fields

Each run summary/record must include:

- `fairness_policy` (`cpu_only_strict` or `mixed_backend_labeled`)
- `thread_policy` (explicit values for OMP/MKL/OPENBLAS/NUMEXPR)
- `runtime_fairness_degraded` (`true`/`false`)
- `effective_threads_observed` (if measurable)
- `device_used`
- `method_backend_path` (native/fallback label)

For benchmark runs through
`documents/experiments/common/run_reference_heldout_scaleout_benchmark.py`,
these fields should be present in machine-readable outputs:

- wrapper-level `scaleout_status.json`
- per-method `runs/<method>/metrics.json` under `results[0].fairness_metadata`

For the formal third-wave scaling round, preparation outputs must also record:

- whether the fixed `100k` artifact is a true `100k` artifact or a
  dataset-ceiling exception
- whether a run belongs to the main panel or supplementary panel

## Execution checklist

1. export fixed thread env vars in launch shell
2. run one quick sanity manifest to confirm env capture fields
3. run formal `10k` and nested `5k` manifests
4. verify fairness fields in metrics/report tables
5. reject runs with missing fairness metadata from paper runtime tables
