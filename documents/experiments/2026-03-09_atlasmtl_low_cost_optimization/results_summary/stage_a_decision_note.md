# Stage A Decision Note

Date: `2026-03-09`

## Current scope of evidence

Completed:

- Stage A `cpu_core`
- Stage A `gpu`
- scheduler check (`adamw_bestwd_plateau`)

## Stage A screening result

Selected `best_wd`:

- `5e-5`

Reason:

- `wd=5e-5` is the strongest practical compromise across CPU and GPU
- it wins both `PHMap` GPU representative points
- it wins `PHMap build_100000_eval10k` on CPU
- it wins `mTCA build_100000_eval10k` on both CPU and GPU
- it stays close to the best non-baseline result on the remaining points except
  `mTCA gpu predict`, where it underperforms

Rejected `weight_decay` candidates:

- `wd=1e-5`
  - too weak on `PHMap`
- `wd=1e-4`
  - too unstable across `PHMap` and `mTCA`

## Scheduler check result

Tested candidate:

- `AdamW + wd=5e-5 + ReduceLROnPlateau`

Decision:

- drop scheduler

Reason:

- plateau underperformed `AdamW + wd=5e-5` on:
  - both `PHMap` CPU points
  - both `PHMap` GPU points
  - `mTCA build_100000_eval10k` on CPU and GPU
  - `mTCA predict_100000_10000` on CPU
- plateau improved only one representative point materially:
  - `mTCA predict_100000_10000` on GPU
- that isolated gain is not enough to justify a scheduler default

## Decision status

Stage A decision:

- lock Stage B candidate to:
  - `AdamW + wd=5e-5`
- stop permanently:
  - `AdamW + wd=5e-5 + ReduceLROnPlateau`
  - any additional `weight_decay` expansion within Stage A

Stage B comparison contract:

- compare `baseline` vs `AdamW + wd=5e-5` only
- do not reinterpret the single `mTCA gpu predict` regression as sufficient
  evidence to reject `wd=5e-5` before Stage B
- use Stage B to test whether `AdamW + wd=5e-5` remains default-acceptable over
  the broader benchmark scope without harming the lightweight positioning of
  `atlasmtl`

## Runtime interpretation constraint

- CPU runs reported `joblib_serial_fallback`, so CPU runtime/resource numbers
  remain provisional
- GPU runs were executed outside the sandbox and did not report runtime
  degradation
