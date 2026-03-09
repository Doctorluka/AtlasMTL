# Formal third-wave HLCA round summary (`2026-03-05`)

## Round conclusion

This formal pilot run is considered successful at process level:

- split materialization (`train10k/test5k`) is reproducible
- CPU group and GPU group both completed with expected method sets
- per-method artifacts (`summary.csv`, `metrics.json`, `scaleout_status.json`) are complete
- native `celltypist` formal training path is now confirmed in the formal flow

Historical-scope note:

- this HLCA run should be treated as a process-valid formal pilot, not the
  final locked-parameter reference for later formal datasets.
- later formal reruns should use the locked defaults from:
  - `documents/protocols/experiment_protocol.md`
  - `documents/experiments/2026-03-05_formal_third_wave_hlca_benchmark/manifests/reference_heldout/*_v2.yaml`

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

4. `scanvi` runtime tuning
- historical note:
  - the HLCA pilot itself used `15/15/10` before the dedicated cross-dataset
    lock experiment was completed.
- current formal default after the dedicated lock experiment:
  - `scvi_max_epochs=25`, `scanvi_max_epochs=25`, `query_max_epochs=20`
  - `n_latent=20`, `batch_size=256`, `datasplitter_num_workers=0`

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
