# atlasmtl

`atlasmtl` is a single-cell reference mapping package with multi-task label
prediction, optional coordinate regression, gated KNN correction, and
Unknown-cell abstention.

## What atlasmtl does

- `AnnData` in, `AnnData` out
- multi-level annotation with a shared representation
- optional coordinate prediction into `X_pred_latent` and `X_pred_umap`
- low-confidence-only KNN correction by default
- optional reliability add-ons:
  - post-hoc calibration
  - open-set scoring
  - hierarchy consistency enforcement

All reliability add-ons are opt-in and preserve the default workflow unless
explicitly enabled.

## Gene ID and feature-space policy

atlasmtl currently aligns genes by exact `adata.var_names` matching. There is
no built-in symbol/Ensembl synonym resolver in the current runtime path, so
formal experiments should standardize gene identifiers before training and
prediction.

Recommended policy:

- use versionless Ensembl IDs as the canonical internal gene namespace
- keep readable symbols in `adata.var["gene_symbol"]`
- record the original `var_names` type and species during preprocessing
- derive the final feature panel after canonicalization

Bundled mapping resource:

- `atlasmtl/resources/gene_id/biomart_human_mouse_rat.tsv.gz`
  - derived from `/home/data/public_data/database/bioMart/GRCh38_Human_Rat_Mouse.txt`
  - renamed to explicit atlasmtl columns for human / mouse / rat mapping

Practical implications:

- `symbol` is easier to read but loses stability across datasets and aliases
- `Ensembl` is less readable but is a safer computation key for exact gene
  alignment
- if raw inputs use Ensembl with version suffixes such as
  `ENSG00000123456.7`, strip the suffix before atlasmtl alignment

For feature selection, prefer a reference-derived HVG panel for formal model
training and benchmark runs. Whole-matrix training should be treated as an
explicit ablation or a resource-rich setting, not the default experimental
protocol.

The intended control surface is:

- preprocessing step
  - declare `var_names_type` and `species`
  - map to versionless Ensembl IDs
  - select the final feature panel
- `build_model()`
  - consume the already-canonicalized matrix

In other words, HVG selection should be configured before or around
`build_model()`, not hidden implicitly inside the core model runtime.

Implemented preprocessing API:

- `atlasmtl.preprocess_reference(...)`
- `atlasmtl.preprocess_query(...)`
- `atlasmtl.PreprocessConfig`
- `atlasmtl.FeaturePanel`
- `atlasmtl.PreprocessReport`

Current defaults:

- canonical internal namespace: versionless Ensembl
- default feature space: `hvg`
- default HVG count: `3000`
- whole matrix must be explicitly requested

## Quick start

```python
import atlasmtl

model = atlasmtl.build_model(
    adata=adata_ref,
    label_columns=["anno_lv1", "anno_lv2", "anno_lv3", "anno_lv4"],
    coord_targets={"latent": "X_ref_latent", "umap": "X_umap"},
    device="auto",
)
model.save("model.pth")

model = atlasmtl.TrainedModel.load("model_manifest.json")
result = atlasmtl.predict(model, adata_query)

adata_query = result.to_adata(adata_query)
labels = result.to_dataframe(mode="minimal")
result.to_csv("predictions.csv", mode="minimal")

model.show_resource_usage()
result.show_resource_usage()
```

Training and prediction also support `show_summary=True`. When
`show_summary=None`, atlasmtl prints the resource summary automatically only in
interactive terminals.

## Python API surface

### `build_model(...)`

Important controls:

- `hidden_sizes`
- `dropout_rate`
- `batch_size`
- `num_epochs`
- `learning_rate`
- `input_transform`
- `coord_targets`
- `val_fraction`
- `early_stopping_patience`
- `early_stopping_min_delta`
- `random_state`
- `preset`
- `calibration_method`
- `domain_key`
- `domain_loss_weight`
- `topology_loss_weight`
- `topology_k`
- `topology_coord`
- `reference_storage`
- `reference_path`
- `num_threads`
- `device`
- `show_progress`
- `show_summary`

Notable defaults:

- `input_transform="binary"`
- `reference_storage="external"`
- `num_threads=10`
- `device="auto"`
- `topology_loss_weight=0.0`
- `domain_loss_weight=0.0`

Current training default remains phmap-consistent:

- `input_transform="binary"` binarizes the input matrix as `(X > 0)`

This default does not decide the gene namespace. Gene-ID canonicalization
should happen before `build_model()`.

### `predict(...)`

Important controls:

- `knn_correction`
- `confidence_high`
- `confidence_low`
- `margin_threshold`
- `knn_k`
- `knn_conf_low`
- `knn_vote_mode`
- `knn_reference_mode`
- `knn_index_mode`
- `input_transform`
- `apply_calibration`
- `openset_method`
- `openset_threshold`
- `openset_label_column`
- `hierarchy_rules`
- `enforce_hierarchy`
- `batch_size`
- `num_threads`
- `device`
- `show_progress`
- `show_summary`

Notable defaults:

- `knn_correction="low_conf_only"`
- `confidence_high=0.7`
- `confidence_low=0.4`
- `margin_threshold=0.2`
- `knn_k=15`
- `knn_conf_low=0.6`
- `knn_vote_mode="majority"`
- `knn_reference_mode="full"`
- `knn_index_mode="exact"`

## Export and writeback

### `PredictionResult`

- `to_adata(adata, mode="minimal" | "standard" | "full")`
- `to_dataframe(mode=...)`
- `to_csv(path, mode=...)`

Writeback behavior:

- `minimal`
  - final labels only, plus `uns["atlasmtl"]` unless disabled
- `standard`
  - final labels, `conf_*`, `margin_*`, and `is_unknown_*`
- `full`
  - all prediction/debug columns, including raw labels, KNN labels, KNN vote
    fraction, and low-confidence flags

Predicted coordinates are never written by default. Set `include_coords=True`
to write `obsm["X_pred_*"]`.

## Artifact layout

`build_model()` defaults to `reference_storage="external"`.

Recommended bundle:

- `model.pth`
- `model_metadata.pkl`
- `model_feature_panel.json` when preprocessing defines a feature panel
- `model_reference.pkl`
- `model_manifest.json`

The manifest is the preferred load entry point for automation. It stores:

- relative artifact paths
- optional `feature_panel_path`
- `reference_storage`
- `input_transform`
- optional SHA-256 `checksums`

External reference storage also supports gzip-compressed `.pkl.gz` files.

Additional run manifests written by the repository:

- `train_run_manifest.json`
- `predict_run_manifest.json`
- benchmark `run_manifest.json`

## CLI support matrix

The Python API is broader than the CLI. The CLI currently exposes a stable
subset.

### Training CLI supports

`scripts/train_atlasmtl.py` currently exposes:

- label columns
- coordinate keys
- encoder/training hyperparameters
- input transform
- validation/early stopping
- reference storage mode
- thread limit
- device selection
- preprocessing controls for gene-ID mapping and feature-space selection

### Prediction CLI supports

`scripts/predict_atlasmtl.py` currently exposes:

- KNN correction mode
- confidence thresholds
- KNN thresholds
- input transform override
- inference batch size
- write mode
- coordinate writeback
- metadata writeback
- thread limit
- device selection
- optional query preprocessing against the model feature panel

### Python-only features today

These are implemented in the Python API but not yet exposed as CLI flags:

- `preset`
- calibration fit controls
- domain alignment controls
- topology loss controls
- KNN vote/reference/index variants
- open-set controls
- hierarchy enforcement controls
- `show_summary`

## Benchmark runner

The benchmark runner now supports atlasmtl plus a first batch of external
comparators for `sc -> sc` reference mapping.

Run:

```bash
/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python benchmark/pipelines/run_benchmark.py \
  --dataset-manifest path/to/dataset_manifest.yaml \
  --output-dir benchmark_out \
  --device cpu
```

Outputs in `--output-dir`:

- `metrics.json`
- `summary.csv`
- optional `summary_by_domain.csv`
- `run_manifest.json`

Current benchmark coverage:

- per-level classification metrics
- coverage and reject metrics
- per-level behavior metrics such as Unknown rate and KNN rescue rate
- calibration metrics when confidence columns are available
- hierarchy metrics when `predict.hierarchy_rules` is provided
- runtime and artifact accounting
- optional coordinate RMSE, trustworthiness, continuity, and neighbor overlap

Implemented methods:

- `atlasmtl`
- `reference_knn`
- `celltypist`
- `scanvi`
- `singler`
- `symphony`
- `azimuth`

Comparator environment notes:

- Python benchmark env:
  `/home/data/fhz/.local/share/mamba/envs/atlasmtl-env`
- Set `NUMBA_CACHE_DIR=/tmp/numba_cache` before running benchmark tests
- Native `Azimuth` / `Seurat v5` R library:
  `/home/data/fhz/seurat_v5`
- Repo-local R library used for comparator additions such as `symphony`:
  `/home/data/fhz/project/phmap_package/atlasmtl/.r_libs`

Comparator implementation notes:

- `celltypist`, `scanvi` run from the Python benchmark env
- `singler` and `symphony` run through R bridge scripts
- `azimuth` prefers the native `Azimuth` path and records
  `implementation_backend` in metadata
- tiny smoke datasets may fall back to Seurat anchor transfer for `azimuth`
  when native Azimuth is numerically unstable; formal benchmarks should use
  native datasets large enough to avoid fallback

Current benchmark limitations:

- the runner accepts one method config per run, not a full comparator matrix
- domain-shift split definitions are not yet standardized
- hierarchy/KNN/topology remain only partially comparable across all external
  methods
- most external comparators are single-level methods, so the fairest direct
  comparison is still a shared target label level

See `documents/protocols/experiment_protocol.md` for the current manifest,
output contract, and comparator benchmark table.

Generate a markdown summary from benchmark output:

```bash
/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python benchmark/reports/generate_markdown_report.py \
  --metrics-json benchmark_out/metrics.json
```

Export paper-oriented CSV + Markdown tables:

```bash
/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python benchmark/reports/export_paper_tables.py \
  --metrics-json benchmark_out/metrics.json
```

## Runtime metadata

Useful runtime fields:

- `model.train_config["resource_summary"]`
- `model.train_config["runtime_summary"]`
- `result.metadata["prediction_runtime"]`
- `result.metadata["calibration_applied"]`
- `result.metadata["knn_vote_mode"]`
- `result.metadata["knn_reference_mode"]`
- `result.metadata["knn_index_mode"]`
- `result.metadata["openset_*"]`
- `result.metadata["hierarchy_*"]`

`uns["atlasmtl"]` receives the prediction metadata by default when using
`to_adata(...)`.

## Development environment

This repository is developed against:

- `/home/data/fhz/.local/share/mamba/envs/atlasmtl-env`

Verified on 2026-02-28:

- Python 3.11.14
- phmap 0.1.1
- torch 2.10.0
- numpy 2.4.2
- pandas 2.3.3
- scanpy 1.11.5
- anndata 0.12.10
- scikit-learn 1.8.0
- numba 0.64.0

Install dev dependencies:

```bash
/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/pip install -e ".[dev]"
```

If `scanpy` fails because numba cannot create its cache, set:

```bash
export NUMBA_CACHE_DIR=/tmp/numba_cache
```

## Repository layout

- `atlasmtl/`
  - package source code
- `benchmark/`
  - benchmark pipelines and reports
- `documents/`
  - design docs, protocol notes, and changelogs
- `notebooks/`
  - reproducible examples
- `vendor/phmap_snapshot/`
  - read-only reference snapshot
