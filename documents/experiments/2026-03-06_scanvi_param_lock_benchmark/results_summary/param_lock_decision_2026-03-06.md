# Scanvi parameter lock decision (`2026-03-06`)

## Decision target

Lock one default `scanvi` parameter set for subsequent formal benchmark runs.

## Candidate pool

Top-2 from stage A:

- `p8_e25_25_20_nl20`
- `p6_e20_20_20_nl20`

## Evidence summary

From `stage_a_param_ranking.csv`:

- `p8_e25_25_20_nl20`
  - mean `macro_f1`: `0.7459`
  - mean `accuracy`: `0.8402`
  - mean total elapsed: `41.24 s`
- `p6_e20_20_20_nl20`
  - mean `macro_f1`: `0.7261`
  - mean `accuracy`: `0.8391`
  - mean total elapsed: `34.41 s`

From `stage_b_stability.csv`:

- at `query_size=5k`, `p8` outperforms `p6` in both mean `macro_f1` and mean `accuracy`
- at `query_size=10k`, `p6` has slightly higher mean `macro_f1`, but `p8` has slightly higher mean `accuracy`

## Final lock

Default lock: `p8_e25_25_20_nl20`

- `scvi_max_epochs=25`
- `scanvi_max_epochs=25`
- `query_max_epochs=20`
- `n_latent=20`
- `batch_size=256`
- `datasplitter_num_workers=0`

## Rationale

- highest cross-dataset stage-A `macro_f1` and `accuracy`
- stage-B confirms stable behavior under seed and query-size perturbations
- no run failures across full stage-A/stage-B schedules
