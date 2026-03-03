# DISCO first-wave preparation execution report

Date: `2026-03-03`

## Scope

This run materialized the first-wave `reference_heldout` preparation assets for
`DISCO_hPBMCs` under the trial `5k` reference build / `1k` heldout predict
design.

## Inputs

- source dataset:
  `/home/data/fhz/project/phmap_package/data/real_test/ProjectSVR/reference_atlas/DISCO_hPBMCs.h5ad`
- manifest:
  `documents/experiments/2026-03-03_projectsvr_disco_benchmark/manifests/reference_heldout/DISCO_hPBMCs__cell_subtype__group_split_v1.yaml`
- split key: `sample`
- domain key: `sample`
- target label: `cell_subtype`
- split strategy: `A+` candidate search (`n_candidates=128`, `seed=2026`)

## Outputs

- prepared reference:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-03/reference_heldout/DISCO_hPBMCs/prepared/group_split_v1_train5k_test1k/reference_train.h5ad`
- prepared heldout query:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-03/reference_heldout/DISCO_hPBMCs/prepared/group_split_v1_train5k_test1k/heldout_test.h5ad`
- feature panel:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-03/reference_heldout/DISCO_hPBMCs/prepared/group_split_v1_train5k_test1k/feature_panel.json`
- split summary:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-03/reference_heldout/DISCO_hPBMCs/prepared/group_split_v1_train5k_test1k/split_summary.json`
- preprocessing summary:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-03/reference_heldout/DISCO_hPBMCs/prepared/group_split_v1_train5k_test1k/preprocessing_summary.json`

## Result summary

- counts contract passed by validating count-like `adata.X`
- preprocessing promoted `adata.X` into `layers["counts"]`
- gene IDs were canonicalized through the bundled symbol-to-Ensembl mapping
  table
- genes: `33538 -> 22500` canonical Ensembl IDs
- feature panel: `3000` HVGs
- query feature alignment: `3000 / 3000` matched, `0` missing
- prepared reference shape: `5000 × 3000`
- prepared heldout shape: `1000 × 3000`

## Split summary

- build pool cells: `83378`
- predict pool cells: `84216`
- build groups: `52`
- predict groups: `48`
- build subset labels retained: `23`
- predict subset labels retained: `23`
- minimum label support after subset sampling:
  - build subset: `2`
  - predict subset: `2`

## Discussion

- The intended DISCO validation path now works: no upstream counts layer was
  required, and the first-wave preprocessing contract correctly standardized
  `adata.X` into `layers["counts"]`.
- The preparation result is suitable for end-to-end pipeline smoke testing and
  for validating comparator input organization.
- Like PH-Map, the current `5k/1k` subset is still too thin for formal
  per-class claims because the sampled subtype support reaches only `2` cells in
  both build and heldout subsets.
