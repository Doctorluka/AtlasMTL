# Internal Discussion: KNN correction status after formal CPU `X_scANVI` run

This note is for internal guidance only. It records how the current KNN
benchmark should influence AtlasMTL design decisions. It is not a paper-ready
claim document.

## Scope of this discussion

The current KNN discussion is based on a single formal run with:

- reference: `sampled_adata_knn_10000.h5ad`
- query: `sampled_adata_knn_3000.h5ad`
- KNN space: `obsm["X_scANVI"]`
- target level of interest: `anno_lv4`
- AtlasMTL setting: `hvg6000 + binary + phmap task weights`
- KNN variants: `off`, `low_conf_only`, `all`

Formal output directory:

- `~/tmp/atlasmtl_knn_scanvi_eval_20260302_cpu_formal_v2/`

## Main result

The current run does **not** support enabling KNN correction as the default
AtlasMTL inference behavior.

Observed pattern:

- `knn_off` has the best overall `accuracy`
- `knn_low_conf_only` improves `macro_f1`, `balanced_accuracy`, and
  `covered_accuracy`
- `knn_all` improves class-balance-oriented metrics further, but at a higher
  harm cost
- both KNN-enabled modes reduce `coverage`
- both KNN-enabled modes increase `unknown_rate`
- both KNN-enabled modes introduce a non-trivial `knn_harm_rate`

In other words, KNN currently behaves more like a selective tradeoff mechanism
than a reliable net performance booster.

## Interpretation

### 1. What KNN appears to help

The current evidence suggests that KNN is mainly helpful for:

- difficult or low-confidence cells
- minority / hard-to-separate label groups
- improving class-balance-oriented metrics more than overall accuracy

This means KNN is best understood as a rescue-style post-processing layer,
rather than a core source of final accuracy gains.

### 2. What KNN appears to hurt

The current evidence also shows that KNN can:

- overwrite correct MTL predictions
- increase reject / unknown behavior
- lower final overall accuracy

Therefore, if the main project objective is:

- "final label transfer should be as accurate and reliable as possible"

then the current default should remain:

- `knn_correction="off"`

### 3. Why `low_conf_only` is the only variant worth retaining

Among the tested modes:

- `off` is the cleanest and strongest default
- `all` is too aggressive
- `low_conf_only` is the only KNN mode that remains methodologically
  interesting

`low_conf_only` is worth retaining because it tests a narrower and more
plausible hypothesis:

- low-confidence cells might benefit from neighborhood rescue without forcing
  KNN onto the whole dataset

However, even this variant should currently be treated as:

- an ablation
- an optional method-specific secondary analysis
- not a default production or paper-mainline setting

## Design implication for AtlasMTL

Current recommendation:

- keep KNN support in the codebase
- keep KNN benchmark support in the benchmark framework
- do **not** make KNN part of the recommended default inference protocol

Recommended protocol split:

- primary protocol:
  - pure MTL prediction
  - `knn_correction="off"`
- secondary protocol:
  - KNN rescue ablation
  - primarily `knn_correction="low_conf_only"`

## Current project-level decision

Until stronger evidence appears, AtlasMTL should be described as:

- a multi-level MTL reference mapping framework whose main value comes from
  the MTL classifier itself

and **not** as:

- a KNN-corrected mapping framework where KNN is necessary for peak
  performance

This keeps the method narrative focused and avoids overcommitting to a module
whose current empirical support is limited.

## What would be needed to reconsider KNN later

KNN could be reconsidered for stronger promotion only if future runs show one
or more of the following:

- consistent overall accuracy improvement across multiple datasets
- clear gain on biologically important rare classes with acceptable harm rate
- better behavior when using AtlasMTL-native spaces rather than only external
  spaces
- stable gains under fixed protocol settings rather than heavily tuned
  dataset-specific settings

Until then, the current internal guidance is:

- keep KNN implemented
- keep KNN documented
- keep KNN off by default

