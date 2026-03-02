# KNN geometry evaluation (internal discussion)

This document is **internal-only** guidance and is not necessarily intended for
the paper.

## Question

Under the real deployment assumption ("query has only an expression matrix"),
does KNN correction improve label transfer performance when KNN geometry comes
from:

- A) a **predicted coordinate regression head** (trained against reference
  `obsm["X_scANVI"]`, but predicting query coords from expression), vs
- B) the model's **internal latent** geometry (`latent_internal`)?

## Required constraints

- Main benchmark must run on a **no-obsm query copy** derived from the same
  query cells as the `X_scANVI`-available query (strip `obsm`/`obsp`).
- KNN must never consume any precomputed query neighbor graph.
- Do not set `predict.knn_query_obsm_key` in the main benchmark.

## What to compare (per label level, highlight `anno_lv4`)

- Accuracy (includes `Unknown` as incorrect)
- Covered accuracy (excludes `Unknown`)
- Coverage / Unknown rate
- Macro-F1 / balanced accuracy
- KNN behavior: rescue rate, harm rate, change rate, KNN coverage
- Resource usage: elapsed time, peak RSS, peak GPU memory (if any)

## Coordinate diagnostic (A only)

Run a separate diagnostic on the original query with `obsm["X_scANVI"]` to
compute coordinate regression quality metrics (no KNN query override).

Interpretation:

- If coordinate metrics are poor, lack of KNN benefit can be attributed to an
  unreliable predicted geometry.
- If coordinate metrics are strong but label metrics do not improve, the KNN
  correction policy itself is likely the limiting factor (thresholds, Unknown
  logic, vote mode, prototypes, etc.).

## Results summary

### Main benchmark (no-obsm query)

| geometry_mode | knn_variant | accuracy_lv4 | covered_acc_lv4 | coverage_lv4 | unknown_rate_lv4 | macro_f1_lv4 | balanced_acc_lv4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| predicted_scanvi_head | knn_off | 0.7683 | 0.8294 | 0.9263 | 0.0737 | 0.6913 | 0.6589 |
| predicted_scanvi_head | knn_lowconf | 0.6557 | 0.8673 | 0.7560 | 0.2440 | 0.6096 | 0.5281 |
| predicted_scanvi_head | knn_all | 0.0603 | 0.0810 | 0.7447 | 0.2553 | 0.0141 | 0.0154 |
| latent_internal | knn_off | 0.7657 | 0.8201 | 0.9337 | 0.0663 | 0.6728 | 0.6387 |
| latent_internal | knn_lowconf | 0.7527 | 0.8492 | 0.8863 | 0.1137 | 0.6797 | 0.6324 |
| latent_internal | knn_all | 0.7533 | 0.8458 | 0.8907 | 0.1093 | 0.6848 | 0.6414 |

### Coordinate diagnostic (A only)

- `scanvi_continuity = 0.9710`
- `scanvi_trustworthiness = 0.9737`
- `scanvi_rmse = 1.3352`
- `scanvi_neighbor_overlap = 0.2153`

### KNN behavior highlights (`anno_lv4`)

- `predicted_scanvi_head + knn_lowconf`: `knn_coverage = 0.2747`, `rescue = 0.0023`, `harm = 0.1487`
- `predicted_scanvi_head + knn_all`: `knn_coverage = 1.0000`, `rescue = 0.0047`, `harm = 0.7373`
- `latent_internal + knn_lowconf`: `knn_coverage = 0.2773`, `rescue = 0.0110`, `harm = 0.0560`
- `latent_internal + knn_all`: `knn_coverage = 1.0000`, `rescue = 0.0153`, `harm = 0.0623`

## Conclusion (current recommendation)

- Main result: geometry choice matters. The previous conclusion "KNN is useless"
  was too broad because `latent_internal` is much more stable than
  `predicted_scanvi_head`.
- However, under the current settings, neither geometry provides evidence that
  KNN improves end-to-end `accuracy` over `knn_off`.
- `predicted_scanvi_head` is not viable for KNN correction in the current
  implementation. Although the predicted scanvi space has high continuity and
  trustworthiness in the coordinate diagnostic, its KNN label-transfer behavior
  is poor, with extremely high harm rates once KNN is enabled.
- `latent_internal` is the only geometry worth retaining for future KNN
  ablations. It keeps accuracy close to baseline and improves covered accuracy,
  but the gain is still accompanied by lower coverage and higher Unknown rate.
- Default `knn_correction`: keep `off`.
- Keep which KNN modes as ablations: retain `latent_internal + low_conf_only`
  and optionally `latent_internal + all`; drop `predicted_scanvi_head` KNN from
  the near-term default path.
- Next technical question: if predicted scanvi coordinates are geometrically
  reasonable but KNN voting is harmful, the limiting factor is likely label
  neighborhood purity / vote policy / hierarchy interaction rather than raw
  regression failure alone.
