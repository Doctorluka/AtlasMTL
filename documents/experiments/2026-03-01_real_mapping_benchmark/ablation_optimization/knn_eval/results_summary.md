# AtlasMTL KNN correction evaluation (real mapping smoke dataset)

Goal: quantify whether AtlasMTL's KNN correction improves label-transfer quality (accuracy + abstention behavior) under the repo's primary positioning: **reliable sc→sc reference mapping and multi-level label transfer**.

This ablation evaluates `knn_correction ∈ {off, low_conf_only, all}` while holding the rest of the training/prediction protocol fixed.

## Repro scripts

- Script: `documents/experiments/2026-03-01_real_mapping_benchmark/ablation_optimization/scripts/run_atlasmtl_knn_eval.py`
- Base manifest: `documents/experiments/2026-03-01_real_mapping_benchmark/ablation_optimization/manifests/atlasmtl_knn_eval_base.yaml`

## Expected outputs

The script writes:

- `metrics.json` (aggregated, paper-table friendly)
- `paper_tables/` (CSV tables)
- `benchmark_report.md` (markdown report)
- `generated_manifests/` (fully expanded per-variant manifests)
- `runs/` (per-variant benchmark runner outputs)

## Key evaluation fields (interpretation)

Primary (per label level):

- `accuracy`, `balanced_accuracy`, `macro_f1`
- `coverage`, `covered_accuracy`, `reject_rate`

KNN behavior (per label level):

- `knn_coverage`
- `knn_change_rate`
- `knn_rescue_rate`, `knn_harm_rate` (and `*_among_used`)

Resource:

- `train_usage.*`, `predict_usage.*` (elapsed time, peak RSS, peak GPU memory, device)

## Notes

- KNN is evaluated in the same latent space used for prediction. When no explicit coordinate heads are trained, AtlasMTL stores and uses the **internal encoder latent** (`knn_space_used="latent_internal"`).
- This dossier is a *smoke* benchmark; do not treat results as paper-final without replicating across datasets and reporting variability.

