# Multi-Level Decision Note

This round supports the claim that AtlasMTL can produce stable multi-level
outputs with perfect observed path consistency under hierarchy enforcement, but
it does not support the stronger claim that the multi-level hierarchy-aware
setting is uniformly superior to the retained single-level formal benchmark on
the finest label level.

## Core conclusions

- KNN is off for all runs in this round.
- All planned runs completed successfully: `16/16`.
- CPU runtime evidence is degraded for fairness analysis because all CPU runs
  triggered `joblib` serial fallback in the restricted environment.
- GPU runs completed cleanly and provide the stronger runtime-quality evidence.
- All runs achieved:
  - `mean_path_consistency_rate = 1.0`
  - `min_path_consistency_rate = 1.0`

## Interpretation against prior single-level formal benchmark

Comparing finest-level AtlasMTL metrics from this round against the retained
third-wave single-level AtlasMTL rows:

- mean `delta_macro_f1` across the 16 matched rows: `-0.007650`
- positive finest-level GPU deltas: `2/8`

This indicates that the current hierarchy-aware multi-level setting is not a
strictly beneficial replacement for the single-level formal benchmark when the
evaluation target is restricted to finest-level headline performance.

## Dataset-level reading

- `HLCA_Core`
  - near parity overall
  - GPU build is slightly positive on finest-level `macro_f1`
- `mTCA`
  - near parity overall
  - CPU build is slightly positive on finest-level `macro_f1`
- `PHMap_Lung_Full_v43_light`
  - the hardest dataset in this round
  - finest-level and full-path performance remain clearly below the stronger
    datasets
- `DISCO_hPBMCs`
  - shallow hierarchy behaves well overall
  - GPU build is positive, but GPU predict remains weaker than the retained
    single-level baseline

## Practical reading

The current hierarchy-aware configuration appears useful for enforcing coherent
multi-level outputs, but it likely trades away some finest-level headline
performance. This makes it suitable for method-specific multi-level reporting,
but not yet for replacing retained single-level manuscript comparison rows.
