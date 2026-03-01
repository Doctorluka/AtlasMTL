# PH-Map vs AtlasMTL Gap Analysis (2026-03-01)

This note explains why the current sampled AtlasMTL run can show lower
single-level `anno_lv4` accuracy than the older PH-Map benchmark notebook at
`/home/data/fhz/project/phmap_package/scripts/05_benchmark_label_trans.ipynb`.

## 1. What the PH-Map notebook reports

From the notebook outputs:

- `PH-Map` sampled `10000` result on `lv4`:
  `0.8361`
- `Celltypist` sampled `10000` result on `lv4`:
  `0.7034`
- `scANVI` sampled `10000` result on `lv4`:
  `0.8054`

Those values come from the notebook's own experimental setup and should not be
assumed to be directly identical to the current AtlasMTL sampled benchmark
protocol.

## 2. Current AtlasMTL sampled result

In the current sampled benchmark:

- single-level `atlasmtl` on `anno_lv4`:
  `0.7467`
- multi-level `atlasmtl` on `anno_lv4`:
  `0.7437`

This is lower than the notebook's PH-Map `lv4` value around `0.8361`.

## 3. Most likely reasons for the gap

### A. Protocol mismatch, not necessarily model regression

The most important caution is that the two runs are not yet strictly
apples-to-apples.

The current AtlasMTL run differs from the old PH-Map benchmark in several
likely ways:

- bundled BioMart-based `symbol -> Ensembl` canonicalization is now applied
- `467` unmapped genes are dropped during canonicalization
- the benchmark is now routed through a unified manifest/reporting layer
- the current sampled run uses the current AtlasMTL defaults rather than the
  exact PH-Map notebook settings

Any one of these can reduce apparent accuracy without indicating that the
core model family is intrinsically worse.

### B. Feature-space change after canonicalization

Current preprocessing changes the effective feature space:

- raw input genes:
  `21977`
- canonical genes retained:
  `21510`

So AtlasMTL is not training on exactly the same symbolic feature space used by
the old PH-Map notebook run. Losing `467` unmapped genes may have a modest but
real effect, especially for fine-grained `lv4` discrimination.

### C. Multi-level objective can trade some flat `lv4` accuracy for structure

AtlasMTL is optimized as a multi-level method. Even when evaluated on the
shared `anno_lv4` label, it is designed to support:

- simultaneous `anno_lv1~4` prediction
- hierarchy-aware consistency
- future KNN rescue / abstention behaviors

That means AtlasMTL is not identical in objective to the older PH-Map framing
of largely unconstrained per-level prediction. A slight drop in flat `lv4`
accuracy may be exchanged for stronger structured behavior.

### D. Current hyperparameters are still conservative

The sampled AtlasMTL runs used a light configuration:

- `hidden_sizes=[256, 128]`
- `dropout_rate=0.2`
- `num_epochs=8`
- `input_transform="binary"`
- CPU execution

This is intentionally closure-oriented, not yet a tuned formal run. The old
PH-Map notebook may reflect a more favorable or more specifically tuned setup.

### E. Binary transform remains a likely bottleneck

AtlasMTL still inherits the PH-Map-style default:

- `input_transform="binary"`

For fine-grained `lv4` labels, this may throw away too much expression
magnitude information. This is a strong candidate explanation for why current
fine-level accuracy lags behind a better-tuned reference result.

## 4. Practical improvement directions

The best next investigations are:

1. strict apples-to-apples rerun against PH-Map
   - same sampled split
   - same feature space
   - same transform
   - same evaluation definition
2. input-transform ablation
   - compare `binary` vs count-aware or expression-preserving transforms
3. training-depth ablation
   - increase epochs
   - compare hidden sizes
   - compare regularization strength
4. whole vs HVG formal ablation
   - current sampled run uses `whole`
   - verify whether a reference-derived HVG panel improves lv4 robustness
5. CPU vs GPU benchmark variants
   - not mainly for accuracy
   - but important for runtime/resource conclusions

## 5. Current judgment

The current evidence does not support the claim that AtlasMTL is definitively
worse than PH-Map.

The more defensible interpretation is:

- AtlasMTL currently closes the full benchmark and multi-level framework
- the sampled run shows competitive but not yet optimized `anno_lv4` accuracy
- there is clear room for accuracy recovery through protocol matching and
  hyperparameter/transform ablation

So the right next move is not to weaken the framework, but to run a controlled
PH-Map-aligned ablation and identify where the loss is coming from.
