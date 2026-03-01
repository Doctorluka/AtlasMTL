# AtlasMTL Ablation Optimization (2026-03-01 follow-up)

This sub-dossier tracks the next AtlasMTL-only benchmark round that is meant to
explain the current gap versus PH-Map and quantify CPU/GPU and resource
tradeoffs.

Scope:

- AtlasMTL only; reuse the existing all-method comparator bundle as baseline
- multi-level runs on `anno_lv1` to `anno_lv4`
- explicit ablations over:
  - `task_weights`: `uniform` vs `phmap`
  - `feature_space`: `whole`, `hvg3000`, `hvg6000`
  - `input_transform`: `binary`, `float`
  - `device`: `cpu`, plus `cuda` only after a gate check passes

Storage split:

- repo-tracked plans, manifests, and scripts live here
- raw runtime artifacts should be written under
  `~/tmp/atlasmtl_real_mapping_benchmark_20260301/ablation_optimization/`

Path rule:

- repo-tracked manifests should prefer atlasmtl-root-relative paths such as
  `data/test_adatas/...`
- run-generated manifests and runtime metadata may contain absolute paths for
  exact reproducibility of a specific run

Directory layout:

- `plan/`
  - locked execution plan for this ablation round
- `manifests/`
  - reusable base manifests
- `scripts/`
  - CUDA gate and ablation runner scripts
- `notes/`
  - discussion notes and environment-specific caveats
