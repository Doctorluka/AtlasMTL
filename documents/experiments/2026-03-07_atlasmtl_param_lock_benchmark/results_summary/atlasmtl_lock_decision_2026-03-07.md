# AtlasMTL lock decision (`2026-03-07`)

## Decision rule

Per device track (`cpu` / `cuda`):

- primary: cross-dataset mean `macro_f1`
- secondary: cross-dataset mean `accuracy`
- tiebreak (`macro_f1` gap `<0.5%`): lower total elapsed

## Locked defaults

- CPU default:
  - `param_id=c5_lr3e4_h256_128_b128`
  - `learning_rate=3e-4`
  - `hidden_sizes=[256,128]`
  - `batch_size=128`
- GPU default:
  - `param_id=g6_lr1e3_h1024_512_b512`
  - `learning_rate=1e-3`
  - `hidden_sizes=[1024,512]`
  - `batch_size=512`

See:

- `atlasmtl_locked_defaults.json`
- `stage_a_core_ranking_cpu.csv`
- `stage_a_core_ranking_gpu.csv`
- `stage_b_stability_cpu.csv`
- `stage_b_stability_gpu.csv`
