# HLCA second-wave scale-out preparation record (`100k/10k/5k`)

- date: `2026-03-05`
- stage: `preparation`
- dataset: `HLCA_Core`
- source h5ad:
  - `/home/data/fhz/project/phmap_package/data/real_test/HLCA/hlca_clean.h5ad`
- command:
  - `documents/experiments/common/prepare_reference_heldout_scaleout.py --prep-manifest documents/experiments/2026-03-03_hlca_benchmark/manifests/reference_heldout/HLCA_Core__ann_level_5__scaleout_prep_v1.yaml --output-dir /home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/prepared/group_split_v2_train100k_test10k_nested5k --build-ceiling 100000 --predict-size-10k 10000 --predict-size-5k 5000 --seed 2026`

## Outputs confirmed

- `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/prepared/group_split_v2_train100k_test10k_nested5k/reference_train.h5ad`
- `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/prepared/group_split_v2_train100k_test10k_nested5k/heldout_test_10k.h5ad`
- `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/prepared/group_split_v2_train100k_test10k_nested5k/heldout_test_5k.h5ad`
- `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/prepared/group_split_v2_train100k_test10k_nested5k/feature_panel.json`
- `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/prepared/group_split_v2_train100k_test10k_nested5k/split_plan.json`
- `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/prepared/group_split_v2_train100k_test10k_nested5k/split_summary.json`
- `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/prepared/group_split_v2_train100k_test10k_nested5k/preprocessing_summary.json`
- `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/HLCA_Core/prepared/group_split_v2_train100k_test10k_nested5k/preparation_resource_summary.json`

## Notes

- no hard failure was observed in preparation for HLCA
- downstream benchmark interruption and recovery are tracked in:
  - `documents/experiments/2026-03-03_hlca_benchmark/results_summary/experiment_record_2026-03-05_scaleout_benchmark_10k.md`
