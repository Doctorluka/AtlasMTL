# AtlasMTL pre-formal parameter-lock benchmark plan (`2026-03-07`)

## Goal

Lock stable and reproducible default `atlasmtl` training parameters before the
formal benchmark round, with separate CPU and GPU tracks.

## Fixed decisions

- run both `cpu` and `cuda` tracks
- lock defaults per device track (not one shared setting)
- fixed runtime thread policy:
  - `num_threads=8` for atlasmtl
- fixed training upper bound:
  - `max_epochs=50` with early stopping
- fixed matrix transform:
  - `input_transform=binary`
- enhanced knobs (`domain/topology/calibration`) are excluded from this main
  lock round and will be benchmarked in a later ablation round

## Data contract

- datasets:
  - `PHMap_Lung_Full_v43_light`
  - `DISCO_hPBMCs`
  - `mTCA`
  - `HLCA_Core`
  - `Vento`
- split files:
  - `reference_train_10k.h5ad`
  - `heldout_test_5k.h5ad`
  - `heldout_test_10k.h5ad`

## Main benchmark design

Stage A (core grid screen, `test5k`, seed `2026`):

- CPU grid (`12` combos):
  - `learning_rate`: `{1e-3, 3e-4, 1e-4}`
  - `hidden_sizes`: `{[256,128], [512,256]}`
  - `batch_size`: `{128, 256}`
- GPU grid (`18` combos):
  - `learning_rate`: `{1e-3, 3e-4, 1e-4}`
  - `hidden_sizes`: `{[256,128], [512,256], [1024,512]}`
  - `batch_size`: `{256, 512}`

Stage B (top-2 confirmation, per track):

- choose top-2 from stage A for each device track
- rerun with:
  - seeds `{17, 23}`
  - query sizes `{5k, 10k}`

## Lock rule

Per device track:

- primary: cross-dataset mean `macro_f1`
- secondary: cross-dataset mean `accuracy`
- tiebreak when macro gap `<0.5%`: lower total elapsed time
- failed/OOM configs cannot be locked

## Required outputs

- `stage_a_core_ranking_cpu.csv|.md`
- `stage_a_core_ranking_gpu.csv|.md`
- `stage_b_stability_cpu.csv|.md`
- `stage_b_stability_gpu.csv|.md`
- `atlasmtl_locked_defaults.json`
- `parameter_guide_2026-03-07_atlasmtl.md`
- `experiment_report_2026-03-07_atlasmtl_param_lock.md`
