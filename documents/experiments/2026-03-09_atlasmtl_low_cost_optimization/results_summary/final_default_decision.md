# Final Default Decision

Date: `2026-03-09`

## Recommended decision

Promote `AdamW + weight_decay=5e-5` as the new default training configuration.

## Decision basis

The clean Stage B GPU confirmation provides the primary evidence for promotion.

Across the Stage B GPU representative points, the candidate improved `macro_f1`
on `7/8` points, with only one formal regression on
`PHMap_Lung_Full_v43_light / predict_100000_10000` (`-0.008676`).

The observed GPU-side gains cover:

- `HLCA_Core / build_100000_eval10k`: `+0.014619`
- `HLCA_Core / predict_100000_10000`: `+0.005678`
- `PHMap_Lung_Full_v43_light / build_100000_eval10k`: `+0.016954`
- `mTCA / build_100000_eval10k`: `+0.014338`
- `mTCA / predict_100000_10000`: `+0.021822`
- `DISCO_hPBMCs / build_100000_eval10k`: `+0.034498`
- `DISCO_hPBMCs / predict_100000_10000`: `+0.041895`

No meaningful resource penalty was observed during the Stage B GPU confirmation:

- GPU memory usage remained effectively unchanged
- RSS differences were negligible
- train-time overhead was modest and stayed within an acceptable low-cost range

## Retained limitation

CPU Stage B evidence was mixed and collected under a constrained environment
with `joblib_serial_fallback`; these results should therefore be treated as
supportive but non-decisive.

The `PHMap_Lung_Full_v43_light / predict_100000_10000` GPU regression remains a
documented caveat and should be retained in the decision note rather than
hidden.

## Not promoted

`ReduceLROnPlateau` should not be promoted to the default configuration.
