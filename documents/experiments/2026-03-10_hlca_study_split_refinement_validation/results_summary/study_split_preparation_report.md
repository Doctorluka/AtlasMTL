# HLCA Study-Split Preparation Report

- scenario: `HLCA_Core` study-grouped validation preparation
- source data: `/home/data/fhz/project/phmap_package/data/real_test/HLCA/hlca_clean.h5ad`
- split key: `study`
- output root:
  - `/tmp/atlasmtl_benchmarks/2026-03-10/hlca_study_split_refinement/HLCA_Core/prepared/formal_split_v1`
- chosen build groups:
  - `Jain_Misharin_2021`
  - `Krasnow_2020`
  - `Misharin_2021`
  - `Nawijn_2021`
- chosen heldout groups:
  - `Banovich_Kropski_2020`
  - `Barbry_Leroy_2020`
  - `Lafyatis_Rojas_2019`
  - `Meyer_2019`
  - `Misharin_Budinger_2018`
  - `Seibold_2020`
  - `Teichmann_Meyer_2019`
- resulting prepared assets:
  - `build_scaling/build_100000/reference_train_100000.h5ad`
  - `build_scaling/build_100000/heldout_build_eval_10k.h5ad`
  - `predict_scaling/fixed_build_100000/heldout_predict_10000.h5ad`
  - `split_plan.json`
  - `split_summary.json`
  - `preprocessing_summary.json`
  - `preparation_resource_summary.json`
- key size outcome:
  - selected build pool: `100000`
  - selected heldout total: `60000`
  - fixed build size for validation: `100000`
- retained warnings:
  - `build_pool_has_label_with_lt25_cells`
  - `build_subset_has_label_with_lt25_cells`
  - `predict_subset_has_label_with_lt10_cells`

Interpretation:

- the stricter `study`-grouped validation path is feasible on HLCA
- the split is intended to support dataset-specific weighting confirmation before reranker validation
- this preparation should be treated as a new independent HLCA validation round
