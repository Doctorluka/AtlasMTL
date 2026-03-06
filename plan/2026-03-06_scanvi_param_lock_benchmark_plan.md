# Scanvi pre-formal parameter-lock benchmark plan (`2026-03-06`)

This plan defines a dedicated `scanvi` parameter confirmation benchmark before formal experiments.

## Goal

Produce method-section evidence for a locked default `scanvi` parameter set using:

- multiple dataset sources (`PHMap`, `DISCO_hPBMCs`, `mTCA`, `HLCA_Core`, `Vento`)
- larger parameter coverage
- explicit runtime/resource records

This round remains under `experiment` scope and is not labeled `formal`.

## Fixed scope

- method: `scanvi` only
- runtime: GPU only (`scanvi` CPU tests are excluded in this round)
- data split contract:
  - train: `reference_train_10k.h5ad`
  - predict stage A: `heldout_test_5k.h5ad`
  - predict stage B: `heldout_test_5k.h5ad` and `heldout_test_10k.h5ad`
- datasets:
  - `PHMap_Lung_Full_v43_light`
  - `DISCO_hPBMCs`
  - `mTCA`
  - `HLCA_Core`
  - `Vento`
- matrix contract:
  - `counts_layer=counts`

## Two-stage parameter grid

Stage A (coarse screen, 8 parameter combos):

- epochs combos: `10/10/5`, `15/15/10`, `20/20/20`, `25/25/20`
- latent dims: `n_latent=10` and `20`
- fixed:
  - `batch_size=256`
  - `datasplitter_num_workers=0`
  - seed=`2026`
  - predict=`5k`

Stage B (confirmation, 8 runs):

- top-2 params selected from stage A by cross-dataset ranking
- each top param is rerun under:
  - seeds `{17, 23}`
  - predict sizes `{5k, 10k}`

## Decision rule

- primary score: cross-dataset mean `macro_f1`
- secondary score: cross-dataset mean `accuracy`
- tie-break (if `macro_f1` gap < `0.5%`): lower total runtime
- exclusion: OOM/failure parameter sets cannot become default

## Required outputs

- raw run index and per-run logs
- `sweep_raw_results.csv`
- `stage_a_param_ranking.csv`
- `stage_b_stability.csv`
- `top2_params.json`
- markdown tables for methods writing
- execution and error-fix records under experiment dossier
