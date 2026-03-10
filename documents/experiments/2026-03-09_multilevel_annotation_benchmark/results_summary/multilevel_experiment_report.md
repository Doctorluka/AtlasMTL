# Sixth-Round Multi-Level Annotation Benchmark Report

## Summary

This round evaluated AtlasMTL as a multi-level `sc -> sc` reference mapping
framework across four datasets with explicit hierarchy chains:

- `HLCA_Core`
- `PHMap_Lung_Full_v43_light`
- `DISCO_hPBMCs`
- `mTCA`

The round used AtlasMTL only, with `knn_correction` fixed to `off`, and the
current software default training path:

- `input_transform="binary"`
- `optimizer_name="adamw"`
- `weight_decay=5e-5`
- `scheduler_name=None`

The executed matrix was:

- `4 datasets x 2 tracks x 2 points = 16 runs`

All `16/16` runs completed successfully.

## Execution status

- CPU runs completed: `8/8`
- GPU runs completed: `8/8`
- CPU degraded-runtime flags: `8/8`
- GPU degraded-runtime flags: `0/8`

The CPU degradation flags reflect restricted-environment `joblib` serial
fallback and should be treated as runtime-fairness caveats rather than method
failures.

## Main findings

### 1. Hierarchy consistency was fully preserved

Across all 16 runs:

- `mean_path_consistency_rate = 1.0`
- `min_path_consistency_rate = 1.0`

This is the clearest positive result from the round. Under the current
hierarchy-enforced setting, AtlasMTL produced fully consistent observed
parent-child prediction paths.

### 2. Deep and mid-depth datasets remained usable

The strongest full-path results were observed on `mTCA`, with full-path
accuracy around `0.933` to `0.942`. `HLCA_Core` also remained strong, with
full-path accuracy around `0.855` to `0.864`.

`DISCO_hPBMCs`, despite being shallower, also behaved well, with full-path
accuracy around `0.883` to `0.887`.

### 3. `PHMap_Lung_Full_v43_light` remained the most difficult hierarchy

`PHMap_Lung_Full_v43_light` produced the weakest finest-level and full-path
results in this round:

- full-path accuracy approximately `0.549` to `0.574`
- finest-level `anno_lv4` mean `macro_f1`
  - CPU: `0.617775`
  - GPU: `0.633190`

This dataset should be treated as the main stress case for the hierarchy-aware
multi-level setting.

### 4. Finest-level performance did not uniformly exceed the retained
single-level formal benchmark

When the finest-level rows from this round were aligned to the retained
third-wave single-level AtlasMTL formal rows:

- mean `delta_macro_f1 = -0.007650`
- positive finest-level GPU deltas: `2/8`

This means the multi-level hierarchy-aware configuration is not currently a
strictly beneficial substitute for the single-level formal benchmark when the
comparison target is limited to finest-level headline metrics.

## Dataset-specific reading

### `HLCA_Core`

`HLCA_Core` was close to parity with the retained single-level baseline. GPU
build produced a small positive finest-level `macro_f1` delta, while the other
rows were near-flat or mildly negative. As a five-level dataset, `HLCA_Core`
supports the claim that AtlasMTL can remain usable on deeper hierarchies.

### `PHMap_Lung_Full_v43_light`

`PHMap_Lung_Full_v43_light` was the clearest weak point. All four finest-level
matched rows were below the retained single-level baseline, and full-path
metrics remained the lowest among the four datasets. This dataset should remain
the main cautionary example in expert discussion.

### `DISCO_hPBMCs`

`DISCO_hPBMCs` showed good shallow-hierarchy behavior and clean path metrics.
However, its finest-level comparison to the retained single-level baseline was
mixed: GPU build improved, but GPU predict was materially weaker.

### `mTCA`

`mTCA` was the cleanest positive hierarchy case in this round. Full-path
accuracy remained high, and finest-level performance stayed close to single-
level formal AtlasMTL, with one slight CPU build improvement and otherwise
small differences.

## Preliminary conclusion

This round supports AtlasMTL as a coherent multi-level annotation framework,
especially on `HLCA_Core`, `mTCA`, and `DISCO_hPBMCs`. The strongest evidence is
not a universal finest-level improvement, but the combination of:

- stable coarse-to-fine label quality
- perfect observed path consistency under hierarchy enforcement
- usable full-path accuracy on multi-level datasets

At the same time, the round does not support the claim that the current
hierarchy-aware multi-level setting should replace the retained single-level
formal benchmark rows for manuscript headline comparison.

## Recommended discussion focus with experts

- Whether the manuscript should present multi-level annotation as a distinct
  method capability rather than as a direct finest-level replacement for the
  retained single-level benchmark
- How to frame `PHMap_Lung_Full_v43_light` as the main hard-case hierarchy
- Whether hierarchy enforcement should remain the default for multi-level
  reporting, or be treated as a configurable post-prediction policy
