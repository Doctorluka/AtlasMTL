# Reference-heldout scale-out execution protocol

Date: `2026-03-04`

This protocol converts the second-wave scale-out plan into concrete execution
artifacts.

## Scope

This round covers only reference-heldout scale-out runs.

Out of scope:

- external query validation
- hierarchical headline benchmark
- final deployment retraining

## Preparation contract

Use `documents/experiments/common/prepare_reference_heldout_scaleout.py` for
all second-wave preparation.

For background execution, prefer a real shell script plus
`documents/experiments/common/launch_background_job.sh` over nested
`nohup zsh -lc "..."` command strings. This makes the shell-expanded command
visible in logs and avoids silent early exits before Python begins.

Locked order:

1. load source `.h5ad`
2. verify or materialize `layers["counts"]`
3. canonicalize gene IDs
4. construct A+ group-aware split
5. materialize `build` and `heldout_10k`
6. sample nested `heldout_5k` from `heldout_10k`
7. derive HVG panel from the build subset
8. align both heldout subsets to the shared panel
9. write prepared assets plus JSON summaries

## Required outputs per dataset

Under `~/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/<dataset>/prepared/<split_name>/`:

- `reference_train.h5ad`
- `heldout_test_10k.h5ad`
- `heldout_test_5k.h5ad`
- `feature_panel.json`
- `split_plan.json`
- `split_summary.json`
- `preprocessing_summary.json`
- `preparation_resource_summary.json`

## Resource monitoring rule

Preparation must emit `preparation_resource_summary.json` with:

- per-phase elapsed seconds
- total elapsed seconds
- average RSS
- peak RSS
- CPU core-equivalent average

Benchmark execution continues to rely on the existing method-level runtime
resource accounting in `metrics.json`, `summary.csv`, and paper tables.

## Dataset roster

| Dataset | Prep manifest | Runtime 10k | Runtime 5k | Split key | Build size |
| --- | --- | --- | --- | --- | ---: |
| `PHMap_Lung_Full_v43_light` | `documents/experiments/2026-03-03_phmap_benchmark/manifests/reference_heldout/PHMap_Lung_Full_v43_light__anno_lv4__scaleout_prep_v1.yaml` | `documents/experiments/2026-03-03_phmap_benchmark/manifests/reference_heldout/PHMap_Lung_Full_v43_light__anno_lv4__scaleout_runtime_10k_v1.yaml` | `documents/experiments/2026-03-03_phmap_benchmark/manifests/reference_heldout/PHMap_Lung_Full_v43_light__anno_lv4__scaleout_runtime_5k_v1.yaml` | `sample` | `100000` |
| `HLCA_Core` | `documents/experiments/2026-03-03_hlca_benchmark/manifests/reference_heldout/HLCA_Core__ann_level_5__scaleout_prep_v1.yaml` | `documents/experiments/2026-03-03_hlca_benchmark/manifests/reference_heldout/HLCA_Core__ann_level_5__scaleout_runtime_10k_v1.yaml` | `documents/experiments/2026-03-03_hlca_benchmark/manifests/reference_heldout/HLCA_Core__ann_level_5__scaleout_runtime_5k_v1.yaml` | `donor_id` | `100000` |
| `mTCA` | `documents/experiments/2026-03-04_projectsvr_mtca_benchmark/manifests/reference_heldout/mTCA__Cell_type_level3__scaleout_prep_v1.yaml` | `documents/experiments/2026-03-04_projectsvr_mtca_benchmark/manifests/reference_heldout/mTCA__Cell_type_level3__scaleout_runtime_10k_v1.yaml` | `documents/experiments/2026-03-04_projectsvr_mtca_benchmark/manifests/reference_heldout/mTCA__Cell_type_level3__scaleout_runtime_5k_v1.yaml` | `orig.ident` | `100000` |
| `DISCO_hPBMCs` | `documents/experiments/2026-03-03_projectsvr_disco_benchmark/manifests/reference_heldout/DISCO_hPBMCs__cell_subtype__scaleout_prep_v1.yaml` | `documents/experiments/2026-03-03_projectsvr_disco_benchmark/manifests/reference_heldout/DISCO_hPBMCs__cell_subtype__scaleout_runtime_10k_v1.yaml` | `documents/experiments/2026-03-03_projectsvr_disco_benchmark/manifests/reference_heldout/DISCO_hPBMCs__cell_subtype__scaleout_runtime_5k_v1.yaml` | `sample` | `100000` |
| `cd4` | `documents/experiments/2026-03-04_projectsvr_cd4_benchmark/manifests/reference_heldout/cd4__cell_subtype__scaleout_prep_v1.yaml` | `documents/experiments/2026-03-04_projectsvr_cd4_benchmark/manifests/reference_heldout/cd4__cell_subtype__scaleout_runtime_10k_v1.yaml` | `documents/experiments/2026-03-04_projectsvr_cd4_benchmark/manifests/reference_heldout/cd4__cell_subtype__scaleout_runtime_5k_v1.yaml` | `sample` | `100000` |
| `cd8` | `documents/experiments/2026-03-04_projectsvr_cd8_benchmark/manifests/reference_heldout/cd8__cell_subtype__scaleout_prep_v1.yaml` | `documents/experiments/2026-03-04_projectsvr_cd8_benchmark/manifests/reference_heldout/cd8__cell_subtype__scaleout_runtime_10k_v1.yaml` | `documents/experiments/2026-03-04_projectsvr_cd8_benchmark/manifests/reference_heldout/cd8__cell_subtype__scaleout_runtime_5k_v1.yaml` | `sample` | `100000` |
| `Vento` | `documents/experiments/2026-03-04_projectsvr_vento_benchmark/manifests/reference_heldout/Vento__annotation__scaleout_prep_v1.yaml` | `documents/experiments/2026-03-04_projectsvr_vento_benchmark/manifests/reference_heldout/Vento__annotation__scaleout_runtime_10k_v1.yaml` | `documents/experiments/2026-03-04_projectsvr_vento_benchmark/manifests/reference_heldout/Vento__annotation__scaleout_runtime_5k_v1.yaml` | `orig.ident` | `50000` |

## Record-keeping rule

Every dataset run must leave two repo-tracked markdown files in the owning
dossier:

- `results_summary/execution_report_<date>.md`
- `results_summary/experiment_record_<date>.md`

The experiment record must capture:

- source input path
- script command
- seed
- chosen split candidate
- warnings
- resource summary path
- errors and fixes
