# AtlasMTL KNN correction evaluation (real mapping dataset with external KNN space)

Goal: quantify whether AtlasMTL's KNN correction improves label-transfer quality (accuracy + abstention behavior) under the repo's primary positioning: **reliable sc→sc reference mapping and multi-level label transfer**.

This ablation evaluates `knn_correction ∈ {off, low_conf_only, all}` while holding the rest of the training/prediction protocol fixed.

## Dataset used in the current formal run

- Reference: `sampled_adata_knn_10000.h5ad` (`10k`)
- Query: `sampled_adata_knn_3000.h5ad` (`3k`)
- KNN space: `obsm["X_scANVI"]` in both reference and query
- Label levels: `anno_lv1`, `anno_lv2`, `anno_lv3`, `anno_lv4`
- Matrix source for AtlasMTL training/prediction: `layers["counts"]`

Repo-local manifest for this run:

- `documents/experiments/2026-03-01_real_mapping_benchmark/ablation_optimization/manifests/atlasmtl_knn_scanvi_space.yaml`

Current formal CPU output directory:

- `~/tmp/atlasmtl_knn_scanvi_eval_20260302_cpu_formal_v2/`

## Relation to the previous no-KNN benchmark

This KNN benchmark is designed as an incremental extension of the previous no-KNN benchmark, not a separate pipeline:

- The main runner remains `benchmark/pipelines/run_benchmark.py`.
- The same label-transfer task is evaluated (`reference -> query`).
- The same label columns are used (`anno_lv1` to `anno_lv4`).
- The same matrix contract is used (`layers["counts"]`).
- The same AtlasMTL preprocessing / HVG / binary-input workflow is kept.
- The only intended experimental change is the addition of a KNN correction branch at prediction time.

## Fairness and interpretation

### What is held constant

For this KNN ablation, the following are intentionally kept aligned with the previous benchmark design:

- AtlasMTL remains the same base predictor.
- Reference/query pairing remains the same style of `sc -> sc` mapping evaluation.
- Primary metrics remain label-transfer metrics rather than integration metrics.
- Resource usage is still recorded in the same benchmark framework.

### What changes in this round

The new element is KNN rescue based on an external precomputed space:

- KNN does not use AtlasMTL's own predicted coordinate head in this run.
- KNN uses `X_scANVI` from `AnnData.obsm` as the neighbor space.
- Therefore the exact interpretation is:
- `AtlasMTL classifier + external-space KNN correction`
- not `AtlasMTL-only latent + KNN correction`

### Fair comparison statement

This means the current round is fair for answering:

- whether adding KNN rescue on top of AtlasMTL can improve final label-transfer behavior under a fixed external neighborhood space

But it is not yet sufficient for claiming:

- AtlasMTL's own learned latent space is itself the source of the KNN gain

Accordingly, this round should be reported as an AtlasMTL-specific secondary analysis / augmentation analysis, not as a replacement for the no-KNN main benchmark.

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
- AtlasMTL still supports the internal-latent fallback path, but that is not the main space used in the current formal run.
- In the current formal run, `knn_space_used` is expected to be `scanvi`.
- The current CPU run is complete and stored under `~/tmp/atlasmtl_knn_scanvi_eval_20260302_cpu_formal_v2/`.
- Internal interpretation and design guidance from this run is recorded in:
  `documents/experiments/2026-03-01_real_mapping_benchmark/ablation_optimization/knn_eval/internal_discussion.md`
- Do not treat this dossier as paper-final without replication across datasets and devices.
