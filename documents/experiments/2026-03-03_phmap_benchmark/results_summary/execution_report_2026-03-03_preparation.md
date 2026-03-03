# PH-Map first-wave preparation execution report

Date: `2026-03-03`

## Scope

This run materialized the first-wave `reference_heldout` preparation assets for
`PHMap_Lung_Full_v43_light` under the trial `5k` reference build / `1k`
heldout predict design.

## Inputs

- source dataset:
  `/home/data/fhz/project/phmap_package/data/real_test/phmap/phmap_clean.h5ad`
- manifest:
  `documents/experiments/2026-03-03_phmap_benchmark/manifests/reference_heldout/PHMap_Lung_Full_v43_light__anno_lv4__group_split_v1.yaml`
- split key: `sample`
- domain key: `sample`
- target label: `anno_lv4`
- split strategy: `A+` candidate search (`n_candidates=128`, `seed=2026`)

## Outputs

- prepared reference:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-03/reference_heldout/PHMap_Lung_Full_v43_light/prepared/group_split_v1_train5k_test1k/reference_train.h5ad`
- prepared heldout query:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-03/reference_heldout/PHMap_Lung_Full_v43_light/prepared/group_split_v1_train5k_test1k/heldout_test.h5ad`
- feature panel:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-03/reference_heldout/PHMap_Lung_Full_v43_light/prepared/group_split_v1_train5k_test1k/feature_panel.json`
- split summary:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-03/reference_heldout/PHMap_Lung_Full_v43_light/prepared/group_split_v1_train5k_test1k/split_summary.json`
- preprocessing summary:
  `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-03/reference_heldout/PHMap_Lung_Full_v43_light/prepared/group_split_v1_train5k_test1k/preprocessing_summary.json`

## Result summary

- counts contract passed via existing `layers["counts"]`
- `adata.X` was not used as the formal counts source
- gene IDs were canonicalized from `var["ENSEMBL"]` after fixing missing-value
  handling in explicit Ensembl columns
- genes: `21977 -> 21510` canonical Ensembl IDs
- feature panel: `3000` HVGs
- query feature alignment: `3000 / 3000` matched, `0` missing
- prepared reference shape: `5000 × 3000`
- prepared heldout shape: `1000 × 3000`

## Split summary

- build pool cells: `117021`
- predict pool cells: `117369`
- build groups: `34`
- predict groups: `30`
- build subset labels retained: `53`
- predict subset labels retained: `55`
- minimum label support after subset sampling:
  - build subset: `4`
  - predict subset: `1`

## Discussion

- The preparation path is now technically correct for PH-Map:
  counts came from `layers["counts"]`, and canonical Ensembl IDs came from the
  dataset’s explicit `ENSEMBL` column rather than an unnecessary symbol-mapping
  fallback.
- The current `5k/1k` random subset is sufficient for flow-through testing, but
  it is not yet suitable for formal per-class reporting because the heldout
  subset contains labels with only `1` cell.
- The immediate next use of this asset should be pipeline smoke validation and
  output organization, not paper-grade accuracy interpretation.
