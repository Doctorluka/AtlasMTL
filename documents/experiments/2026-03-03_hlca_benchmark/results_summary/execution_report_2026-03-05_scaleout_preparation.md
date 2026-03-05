# HLCA second-wave scale-out preparation (`100k/10k/5k`)

- scenario: `HLCA_Core` reference-heldout preparation for second-wave scale-out
- manifest:
  - `documents/experiments/2026-03-03_hlca_benchmark/manifests/reference_heldout/HLCA_Core__ann_level_5__scaleout_prep_v1.yaml`
- output root:
  - `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/prepared/group_split_v2_train100k_test10k_nested5k/`

## Split materialization summary

- split key / domain key: `donor_id` / `donor_id`
- target label: `ann_level_5`
- build pool: `291562` cells -> build subset: `100000` cells
- predict pool: `293382` cells -> heldout `10k`: `10000` cells
- nested heldout `5k`: `5000` cells from `10k` heldout
- split warnings:
  - `build_pool_has_label_with_lt25_cells`
  - `build_subset_has_label_with_lt25_cells`
  - `predict_subset_has_label_with_lt10_cells`

## Preprocessing contract summary

- species: `human`
- canonical gene space: Ensembl (`var_names_type=ensembl`)
- input matrix type declared/detected: `lognorm` / `lognorm`
- counts layer used: `layers["counts"]`
- counts check result: `counts_confirmed`
- feature space: `hvg` with `hvg_method=seurat_v3`, `n_top_genes=3000`
- counts and canonicalization checks passed without preprocessing warnings

## Preparation resource summary

- phase: `prepare_reference_heldout_scaleout`
- elapsed: `382.3883 s`
- avg RSS: `64.0343 GB`
- peak RSS: `86.7406 GB`
- device: `cpu`
- throughput: `1529.7122 cells/s` (over `584944` items)

## Key machine-truth files

- `split_summary.json`
- `preprocessing_summary.json`
- `preparation_resource_summary.json`
- `reference_train.h5ad`
- `heldout_test_10k.h5ad`
- `heldout_test_5k.h5ad`
