# Phase-1 preparation report

Date: `2026-03-06`

This report summarizes the completed phase-1 preparation results for the formal
third-wave scaling round.

## Scope

Completed in phase 1:

- data audit
- group-aware split generation
- prepared build-scaling subsets
- prepared predict-scaling subsets
- dataset ceiling summaries
- preparation resource summaries

Main-panel datasets:

- `HLCA_Core`
- `PHMap_Lung_Full_v43_light`
- `mTCA`
- `DISCO_hPBMCs`

Supplementary dataset:

- `Vento`

## Ceiling summary

| Dataset | Panel | Feasible build grid | Fixed build for predict scaling | Feasible predict grid | Predict pool |
| --- | --- | --- | ---: | --- | ---: |
| `HLCA_Core` | `main` | `10k, 20k, 30k, 50k, 100k, 150k, 200k, 300k` | `100k` | `1k, 3k, 5k, 8k, 10k, 15k, 20k, 50k` | `50k` |
| `PHMap_Lung_Full_v43_light` | `main` | `10k, 20k, 30k, 50k, 100k, 150k` | `100k` | `1k, 3k, 5k, 8k, 10k, 15k, 20k, 50k` | `50k` |
| `mTCA` | `main` | `10k, 20k, 30k, 50k, 100k` | `100k` | `1k, 3k, 5k, 8k, 10k, 15k, 20k, 50k` | `50k` |
| `DISCO_hPBMCs` | `main` | `10k, 20k, 30k, 50k, 100k` | `100k` | `1k, 3k, 5k, 8k, 10k, 15k, 20k, 50k` | `50k` |
| `Vento` | `supplementary` | `10k, 20k, 30k, 50k` | `50k` | `1k, 3k, 5k, 8k, 10k` | `10k` |

## Preparation output roots

Main-panel prepared roots:

- `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/HLCA_Core/prepared/formal_split_v1/`
- `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/PHMap_Lung_Full_v43_light/prepared/formal_split_v1/`
- `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/mTCA/prepared/formal_split_v1/`
- `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/DISCO_hPBMCs/prepared/formal_split_v1/`

Supplementary prepared root:

- `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/Vento/prepared/formal_split_v1/`

## Required machine-readable files confirmed

Each dataset now has:

- `split_plan.json`
- `split_summary.json`
- `preprocessing_summary.json`
- `preparation_resource_summary.json`
- `dataset_ceiling_summary.json`

Each dataset also has:

- build-scaling prepared subsets under `build_scaling/`
- predict-scaling prepared subsets under `predict_scaling/`
- canonical split snapshots under `canonical_subsets/`

## Important correction recorded

`Vento` required one correction during phase 1.

Root cause:

- dataset-level empty `predict_tail_optional: []` was incorrectly treated as a
  fallback to the global default tail

Effect:

- `Vento` was initially forced into an invalid larger heldout contract and its
  build ceiling collapsed to `10k`

Resolution:

- fix list-override handling in the preparation script
- delete the incorrect tmp output
- rerun `Vento`

Final corrected `Vento` ceiling is now:

- build: `10k / 20k / 30k / 50k`
- fixed build: `50k`
- predict: `1k / 3k / 5k / 8k / 10k`

## Manifest generation

Formal benchmark manifests are generated into:

- `documents/experiments/2026-03-06_formal_third_wave_scaling/manifests/reference_heldout/`

Manifest classes generated:

- `CPU` build-scaling manifests
- `GPU` build-scaling manifests
- `CPU` predict-scaling manifests
- `GPU` predict-scaling manifests

The manifest generator reuses:

- phase-1 prepared assets
- locked AtlasMTL defaults
- locked scANVI defaults
- explicit `celltypist` formal backend config

## Next step

Phase 2 should start from these prepared assets and generated manifests:

1. run build scaling by dataset and device group
2. run predict scaling by dataset and device group
3. export performance/resource/protocol tables from the formal outputs
