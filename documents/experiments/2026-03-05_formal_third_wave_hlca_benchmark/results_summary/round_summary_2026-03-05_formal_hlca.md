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
| atlasmtl (cuda) | 0.8536 | 0.5658 | 1.4069 | 0.0288 | 0.0351 | 0.0229 |
| scanvi (cuda) | 0.8672 | 0.6422 | 36.5337 | 7.0884 | 0.1072 | 0.1010 |

### Interpretation

- GPU path is functional and stable in a proper host shell.
- `scanvi` remains the heaviest GPU method by elapsed time in this setup.
- `atlasmtl` GPU path is fast at both train and predict under current settings.

## Optimization and micro-tuning targets

1. Resource fairness metadata
- add explicit `fairness_policy/thread_policy/runtime_fairness_degraded/effective_threads_observed` into machine-readable run payloads (not only report text).

2. R comparator resource visibility
- status update: peak RSS now captured for `singler/symphony/seurat_anchor_transfer`
  via subprocess `/usr/bin/time` fallback.
- remaining gap: effective CPU core-equivalent still limited for R comparators in
  this environment.

3. GPU reliability gate
- add a preflight CUDA check in the GPU launcher:
  - fail fast with clear message if `torch.cuda.is_available()` is false.

4. `scanvi` runtime tuning (without changing benchmark scope)
- completed mini ablation on `HLCA train10k/test5k`:
  - `20/20/20`: accuracy `0.8516`, macro_f1 `0.6881`, train+predict `63.28s`
  - `15/15/10`: accuracy `0.8708`, macro_f1 `0.6728`, train+predict `43.97s`
  - `10/10/5`: accuracy `0.8676`, macro_f1 `0.6198`, train+predict `28.13s`
- locked wave-3 default for scale-out runtime runs:
  - `scvi_max_epochs=15`, `scanvi_max_epochs=15`, `query_max_epochs=10`
  - `datasplitter_num_workers=0` (deterministic and lightweight process behavior)

5. CPU thread policy consistency
- keep fixed thread env vars and add per-method recorded `num_threads_used` where available, especially for Python comparators.
  
6. Wrapper efficiency for mixed method sets
- status update: wrapper now prepares/trains CellTypist only when `celltypist` is
  included in `--methods`; GPU group no longer pays this overhead.

## Next action for wave-3 expansion

- replicate this exact template to `PHMap`, `DISCO`, `mTCA`, `Vento`.
- keep method grouping unchanged:
  - CPU group excludes `scanvi`
  - GPU group contains `atlasmtl + scanvi`
