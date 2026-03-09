# Final Default Decision

Date: `2026-03-09`

## Candidate under confirmation

- baseline: `Adam`, `weight_decay=0`
- candidate: `AdamW + wd=5e-5`

## Stage B evidence summary

CPU (`cpu_core`):

- completed with `joblib_serial_fallback` on every point
- candidate improved `3/8` representative points
- candidate underperformed `5/8` representative points
- CPU fairness degradation means these runtime results remain provisional

GPU (`gpu`):

- completed outside the sandbox without fairness degradation
- candidate improved `7/8` representative points
- candidate underperformed `1/8` representative point:
  - `PHMap_Lung_Full_v43_light / predict_100000_10000` (`-0.008676` macro-F1)
- candidate improved the previously watched `mTCA / predict_100000_10000`
  point (`+0.021822` macro-F1)

Representative GPU gains:

- `DISCO_hPBMCs / build_100000_eval10k`: `+0.034498`
- `DISCO_hPBMCs / predict_100000_10000`: `+0.041895`
- `mTCA / build_100000_eval10k`: `+0.014338`
- `mTCA / predict_100000_10000`: `+0.021822`
- `PHMap_Lung_Full_v43_light / build_100000_eval10k`: `+0.016954`

Resource interpretation:

- no meaningful GPU memory penalty was observed
- RSS differences were negligible
- train-time overhead on GPU was present on some points but remained modest
- CPU runtime comparisons are not decisive because of the restricted execution
  mode

## Decision

- promote `AdamW + wd=5e-5` as the new default training configuration
- keep scheduler disabled by default
- do not reopen the `weight_decay` grid in this round

## Reporting caveat

- when summarizing this round, explicitly note that CPU Stage A and Stage B runs
  were affected by `joblib_serial_fallback`
- use the non-degraded GPU confirmation as the primary fairness-grade evidence
