# AtlasMTL Ablation Optimization Plan

Locked variables for the next round:

- counts source: `layers["counts"]`
- binary: thresholded counts (`>0 -> 1`)
- float: raw counts as `float32`
- no default `normalize_total` or `log1p`
- feature-space grid: `whole`, `hvg3000`, `hvg6000`
- task weights:
  - uniform: `[1.0, 1.0, 1.0, 1.0]`
  - phmap: `[0.3, 0.8, 1.5, 2.0]`
- device:
  - always run CPU
  - run CUDA only if the benchmark-entry gate succeeds

Primary questions:

1. Does PH-Map-style weighting recover `anno_lv4` accuracy?
2. Does `whole` outperform HVG once resources are accounted for?
3. Does binary encoding improve resource efficiency enough to justify any
   accuracy tradeoff?
4. Is AtlasMTL sufficiently CPU-efficient that GPU becomes optional for users?
