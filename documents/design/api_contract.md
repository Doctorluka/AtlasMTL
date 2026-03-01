# atlasmtl API Contract

## Scope

This document defines the stable high-level interfaces that users and internal
tools should rely on:

- Python API:
  - `build_model()`
  - `predict()`
  - `TrainedModel`
  - `PredictionResult`
- CLI:
  - `scripts/train_atlasmtl.py`
  - `scripts/predict_atlasmtl.py`
- Model artifacts:
  - `model.pth`
  - `model_metadata.pkl`
  - `model_reference.pkl`
  - `model_manifest.json`

## Python API

### `build_model(...) -> TrainedModel`

Purpose:
- Train an atlasmtl model from a reference `AnnData`.

Required inputs:
- `adata`
- `label_columns`

Important optional controls:
- `coord_targets`
- `hidden_sizes`
- `dropout_rate`
- `batch_size`
- `num_epochs`
- `learning_rate`
- `input_transform`
- `val_fraction`
- `early_stopping_patience`
- `early_stopping_min_delta`
- `random_state`
- `preset`
- `domain_key`
- `domain_loss_weight`
- `domain_loss_method`
- `topology_loss_weight`
- `topology_k`
- `topology_coord`
- `calibration_method`
- `calibration_max_iter`
- `calibration_lr`
- `reference_storage`
- `reference_path`
- `num_threads`
- `device`
- `show_progress`
- `show_summary`

Coordinate behavior:
- `coord_targets=None` means no-coordinate training
- coordinate heads are enabled only when the user explicitly provides the
  `obsm` mapping to use

Defaults that should remain stable unless a versioned change is announced:
- `input_transform="binary"`
- `reference_storage="external"`
- `num_threads=10`
- `device="auto"`
- `domain_loss_method="mean"`
- `domain_loss_weight=0.0`
- `topology_loss_weight=0.0`
- `topology_k=10`
- `topology_coord="latent"`

### `predict(...) -> PredictionResult`

Purpose:
- Run classification, coordinate prediction, optional KNN correction, and
  closed-loop Unknown assignment on a query `AnnData`.

Important optional controls:
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

Defaults that should remain stable unless a versioned change is announced:
- `knn_correction="low_conf_only"`
- `confidence_high=0.7`
- `confidence_low=0.4`
- `margin_threshold=0.2`
- `knn_k=15`
- `knn_conf_low=0.6`
- `knn_vote_mode="majority"`
- `knn_reference_mode="full"`
- `knn_index_mode="exact"`

## `TrainedModel`

Stable responsibilities:
- hold the trained neural network
- hold label encoders and training gene order
- hold coordinate scaling statistics
- hold or resolve reference data needed for KNN
- hold training metadata including elapsed time, resource summary, and optional
  calibration/domain/topology settings
- expose progress-aware train/predict execution through the public API
- save and load artifact bundles

Stable methods:
- `save(path)`
- `load(path, device=None)`

Load contract:
- `path` may point to `model.pth`
- `path` may point to `model_manifest.json`
- if no manifest is present, legacy filename-based loading remains supported

## `PredictionResult`

Stable responsibilities:
- keep full prediction outputs in memory
- allow selective export without forcing everything into `AnnData`
- retain prediction runtime/resource metadata for benchmark accounting
- retain calibration, KNN, open-set, and hierarchy metadata for export and
  traceability

Stable methods:
- `to_adata(adata, mode="standard", include_coords=False, include_metadata=True)`
- `to_dataframe(mode="standard")`
- `to_csv(path, mode="standard", index=True, **kwargs)`

### Export modes

- `minimal`
  - writes or returns only `pred_<level>`
- `standard`
  - writes or returns `pred_<level>`, `conf_<level>`, `margin_<level>`,
    `is_unknown_<level>`
- `full`
  - writes or returns all prediction/debug columns

### Index contract

- `to_dataframe()` and `to_csv()` preserve `adata.obs_names` as the row index by
  default
- this makes exported tables directly joinable back into `adata.obs`

## AnnData writeback contract

Final prediction labels:
- `obs["pred_<level>"]`

Optional confidence outputs:
- `obs["conf_<level>"]`
- `obs["margin_<level>"]`
- `obs["is_unknown_<level>"]`

Optional debug outputs:
- `obs["pred_<level>_raw"]`
- `obs["pred_<level>_knn"]`
- `obs["used_knn_<level>"]`
- `obs["knn_vote_frac_<level>"]`
- related low-confidence flags

Optional coordinates:
- `obsm["X_pred_latent"]`
- `obsm["X_pred_umap"]`

Metadata:
- `uns["atlasmtl"]`

The metadata payload may contain:

- threshold settings
- `train_config`
- device/thread/runtime summaries
- calibration metadata
- KNN mode metadata
- open-set metadata
- hierarchy metadata

## CLI contract

### `scripts/train_atlasmtl.py`

Purpose:
- thin CLI wrapper over `build_model()`

Required flags:
- `--adata`
- `--out`
- `--labels`

Stable user-facing controls:
- coordinate keys
- encoder/training hyperparameters
- input transform
- thread limit
- device selection
- validation / early stopping
- reference storage mode

Current limitation:

- the CLI intentionally exposes only a stable subset of the Python training API
- `preset`, calibration controls, domain controls, and topology controls are
  currently Python API features, not CLI flags

### `scripts/predict_atlasmtl.py`

Purpose:
- thin CLI wrapper over `TrainedModel.load()`, `predict()`, and
  `PredictionResult.to_adata()`

Required flags:
- `--model`
- `--adata`
- `--out`

Stable user-facing controls:
- KNN mode
- confidence thresholds
- KNN thresholds
- inference batch size
- thread limit
- device selection
- writeback mode
- coordinate writeback
- metadata writeback

Current limitation:

- the CLI intentionally exposes only a stable subset of the Python prediction
  API
- KNN vote/reference/index variants, open-set controls, and hierarchy controls
  currently require the Python API

## Artifact contract

Recommended bundle:
- `model.pth`
- `model_metadata.pkl`
- `model_feature_panel.json` when preprocessing metadata includes a feature panel
- `model_reference.pkl`
- `model_manifest.json`

Current artifact additions:

- `model_manifest.json["checksums"]`
- `model_manifest.json["feature_panel_path"]` when a standalone feature-panel artifact is written
- optional gzip-compressed external reference assets
- train, predict, and benchmark run manifests

Manifest purpose:
- stable entry point for automation
- explicit artifact paths
- schema versioning anchor

## Non-goals of this contract

This document does not freeze:
- internal module boundaries
- helper function names under `core/`, `mapping/`, or `io/`
- internal metadata layout beyond the documented public fields

Those internals may change as long as the interfaces above remain compatible.
