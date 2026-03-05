# Formal third-wave HLCA round summary (`2026-03-05`)

## Round conclusion

This formal pilot run is considered successful at process level:

- split materialization (`train10k/test5k`) is reproducible
- CPU group and GPU group both completed with expected method sets
- per-method artifacts (`summary.csv`, `metrics.json`, `scaleout_status.json`) are complete
- native `celltypist` formal training path is now confirmed in the formal flow

## GPU-focused summary

### What happened

- first GPU attempt failed in restricted execution (CUDA device not visible)
- rerun in CUDA-available shell succeeded for both methods:
  - `atlasmtl (cuda)`
  - `scanvi (cuda)`

### Current GPU results (`ann_level_5`)

| method | accuracy | macro_f1 | train_elapsed_s | predict_elapsed_s | train_peak_gpu_gb | predict_peak_gpu_gb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| atlasmtl (cuda) | 0.8384 | 0.5274 | 2.0867 | 0.0446 | 0.0351 | 0.0229 |
| scanvi (cuda) | 0.8734 | 0.7016 | 54.0555 | 13.8746 | 0.1108 | 0.1023 |

### Interpretation

- GPU path is functional and stable in a proper host shell.
- `scanvi` remains the heaviest GPU method by elapsed time in this setup.
- `atlasmtl` GPU path is fast at both train and predict under current settings.

## Optimization and micro-tuning targets

1. Resource fairness metadata
- add explicit `fairness_policy/thread_policy/runtime_fairness_degraded/effective_threads_observed` into machine-readable run payloads (not only report text).

2. R comparator resource visibility
- extend subprocess monitoring for `singler/symphony/seurat_anchor_transfer` to improve RSS/core-equivalent capture (currently partial/zeroed in this run).

3. GPU reliability gate
- add a preflight CUDA check in the GPU launcher:
  - fail fast with clear message if `torch.cuda.is_available()` is false.

4. `scanvi` runtime tuning (without changing benchmark scope)
- evaluate smaller epoch settings for formal runtime tables (for example controlled `scvi_max_epochs/scanvi_max_epochs/query_max_epochs` ablation) and lock one setting for all datasets in wave 3.

5. CPU thread policy consistency
- keep fixed thread env vars and add per-method recorded `num_threads_used` where available, especially for Python comparators.

## Next action for wave-3 expansion

- replicate this exact template to `PHMap`, `DISCO`, `mTCA`, `Vento`.
- keep method grouping unchanged:
  - CPU group excludes `scanvi`
  - GPU group contains `atlasmtl + scanvi`
