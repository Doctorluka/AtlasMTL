# Implementation Record: 2026-03-09 Multi-Level Annotation Benchmark

## Scope implemented

The sixth-round multi-level AtlasMTL benchmark scaffold was implemented and the
full planned run matrix was executed.

Implemented assets:

- dossier README
- round plan
- dataset hierarchy inventory generator
- hierarchy-rule generator
- multilevel manifest generator
- CPU and GPU execution scripts
- result collector

## Dataset hierarchy contract frozen for this round

- `HLCA_Core`
  - `ann_level_1 -> ann_level_5`
- `PHMap_Lung_Full_v43_light`
  - `anno_lv1 -> anno_lv4`
- `DISCO_hPBMCs`
  - `cell_type -> cell_subtype`
- `mTCA`
  - `Cell_type_level1 -> Cell_type_level3`

## Fixed runtime contract

- `atlasmtl` only
- `knn_correction: off`
- `optimizer_name: adamw`
- `weight_decay: 5e-5`
- `scheduler_name: null`
- `input_transform: binary`
- representative points:
  - `build_100000_eval10k`
  - `predict_100000_10000`
- tracks:
  - `cpu_core`
  - `gpu`

## Smoke checks completed

Executed:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache /home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python documents/experiments/2026-03-09_multilevel_annotation_benchmark/scripts/generate_dataset_hierarchy_inventory.py
NUMBA_CACHE_DIR=/tmp/numba_cache /home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python documents/experiments/2026-03-09_multilevel_annotation_benchmark/scripts/generate_multilevel_manifests.py
bash -n documents/experiments/2026-03-09_multilevel_annotation_benchmark/scripts/run_multilevel_cpu.sh
bash -n documents/experiments/2026-03-09_multilevel_annotation_benchmark/scripts/run_multilevel_gpu.sh
NUMBA_CACHE_DIR=/tmp/numba_cache /home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python documents/experiments/2026-03-09_multilevel_annotation_benchmark/scripts/collect_multilevel_results.py
python -m compileall documents/experiments/2026-03-09_multilevel_annotation_benchmark/scripts
```

Observed outcomes:

- hierarchy inventory generated successfully with `4` dataset rows
- multilevel manifest generation produced exactly `16` manifests
- per-dataset hierarchy JSON files were generated
- CPU/GPU shell scripts passed syntax validation
- collector ran successfully with `0` runs present and wrote empty result tables
- script directory passed compileall

## Execution update

After the scaffold was verified, the full run matrix was executed:

- `4 datasets x 2 tracks x 2 points = 16` runs
- completed successfully: `16/16`
- failed runs: `0`
- CPU degraded-runtime flags: `8/8`
- GPU degraded-runtime flags: `0/8`

Observed execution caveats:

- the original scale-out wrapper could not be reused because it required exactly
  one label column
- the round was switched to direct `benchmark/pipelines/run_benchmark.py`
  invocation for AtlasMTL-only multi-level execution
- an initial manifest revision included unsupported metadata keys and was
  regenerated before the final successful run

Primary generated result files:

- `results_summary/levelwise_performance.csv`
- `results_summary/hierarchy_performance.csv`
- `results_summary/reliability_performance.csv`
- `results_summary/multilevel_summary.md`
- `results_summary/multilevel_decision_note.md`
- `results_summary/multilevel_experiment_report.md`
- `results_summary/hierarchy_aware_discussion_note.md`

## Final status

This round is completed and ready for expert review.
