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

For paper-grade runtime tables, prefer `cpu_only_strict` first.

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

For formal third-wave runs, keep second-wave dataset sizing unless explicitly
overridden and justified:

- default references: `100k build + 10k heldout + nested 5k`
- `Vento`: `50k build + 10k heldout + nested 5k`
- `cd4/cd8`: excluded until raw-count contract is satisfied

Quick debug runs (for pipeline validation only) may downsample training (for
example `train5k`), but must be labeled as `debug_only`.

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

## Execution checklist

1. export fixed thread env vars in launch shell
2. run one quick sanity manifest to confirm env capture fields
3. run formal `10k` and nested `5k` manifests
4. verify fairness fields in metrics/report tables
5. reject runs with missing fairness metadata from paper runtime tables
