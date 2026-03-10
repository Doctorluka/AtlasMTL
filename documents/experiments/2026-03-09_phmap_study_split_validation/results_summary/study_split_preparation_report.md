# PH-Map Study-Split Preparation Report

- scenario: `PHMap_Lung_Full_v43_light` study-grouped validation preparation
- source data: `/home/data/fhz/project/phmap_package/data/real_test/phmap/phmap_clean.h5ad`
- split key: `study`
- output root:
  - `/tmp/atlasmtl_benchmarks/2026-03-09/multilevel_annotation_study_split/PHMap_Lung_Full_v43_light/prepared/formal_split_v1/`
- chosen build groups:
  - `Gayathri_Viswanathan_2023`
  - `Taylor_Adams_2020`
  - `Zhou_2024`
- chosen heldout groups:
  - `Jonas_Schupp_2021`
  - `Slaven_Crnkovic_2022`
  - `Tijana_Tuhy_2025`
- resulting prepared assets:
  - `build_scaling/build_100000/reference_train_100000.h5ad`
  - `build_scaling/build_100000/heldout_build_eval_10k.h5ad`
  - `predict_scaling/fixed_build_100000/heldout_predict_10000.h5ad`
  - `split_plan.json`
  - `split_summary.json`
  - `preprocessing_summary.json`
  - `preparation_resource_summary.json`
- key size outcome:
  - selected build pool: `160922`
  - selected build subset: `150000`
  - selected heldout subset: `60000`
  - fixed build size for predict scaling: `100000`
- warning retained:
  - `build_pool_has_label_with_lt25_cells`

Interpretation:

- the stricter `study`-grouped validation path is feasible on PH-Map
- the study-level split preserves high heldout label coverage while enforcing
  complete study isolation
- this preparation should be treated as a new independent PH-Map validation
  round rather than an extension of the original sample-split sixth round
