# atlasmtl Architecture

## Scope

This document describes the current atlasmtl architecture as implemented in the
repository on 2026-02-28. It is intentionally practical: it explains the
module boundaries, the train/predict data flow, and the metadata and artifact
contracts that matter for users, benchmark scripts, and future roadmap work.

## Module layout

- `atlasmtl/core/`
  - public training, prediction, evaluation, runtime helpers, and typed result
    containers
- `atlasmtl/mapping/`
  - confidence, KNN, calibration, open-set scoring, and hierarchy utilities
- `atlasmtl/models/`
  - artifact serialization, manifest handling, checksums, presets, and
    reference storage helpers
- `atlasmtl/io/`
  - AnnData writeback and prediction-frame export helpers
- `atlasmtl/utils/`
  - progress and runtime monitoring helpers
- `benchmark/`
  - benchmark dataset descriptors, method wrappers, runners, and reports

### Benchmark method layer

The benchmark method layer now has an explicit configuration boundary:

- benchmark dataset manifest
  - global preprocessing contract such as `var_names_type`, `species`,
    `input_matrix_type`, and `counts_layer`
- comparator method config
  - method-specific overrides such as `target_label_column`,
    `reference_layer`, `query_layer`, or `counts_layer`

The intended precedence is:

1. method-specific `reference_layer` / `query_layer`
2. method-specific `counts_layer`
3. benchmark-manifest `counts_layer`
4. final fallback: `counts`

The public Python API remains intentionally thin:

- `build_model()`
- `predict()`
- `TrainedModel`
- `PredictionResult`

These are re-exported from `atlasmtl/core/api.py`.

## Core semantics

atlasmtl should be read as a label-transfer system with supporting structure and
context, not as four unrelated prediction targets.

### `label`

`label` is the primary output and the primary research objective.

- final outputs are `obs["pred_<level>"]`
- the main question is whether atlasmtl can transfer labels accurately and
  reliably
- benchmark priority therefore starts from classification, calibration,
  abstention, and robustness metrics

### `hierarchy`

`hierarchy` is a structural constraint on labels rather than a separate task.

- it checks whether coarse-to-fine label paths are self-consistent
- it does not replace per-level labels
- it acts as an additional validity layer on top of label prediction

In practice, hierarchy is currently enforced during post-processing and
evaluated through path-consistency metrics.

### `coordinate`

`coordinate` is a supporting representation rather than the primary endpoint.

- it places query cells into a reference-aligned coordinate system
- it primarily supports KNN correction and atlas-aligned interpretation
- it provides the geometry used by KNN rescue when coordinate heads are
  available
- it is not intended to be a co-equal primary task alongside label prediction

Coordinate quality matters because it affects rescue behavior and atlas reuse,
but it is not the main success criterion of the project.

The internal priority should be read as:

1. `latent` coordinates are the main correction space when available
2. `umap` coordinates are primarily a visualization and interpretation space
3. both remain auxiliary to the label-transfer objective

### `domain`

`domain` is contextual metadata describing where a cell comes from rather than
what it is.

Typical examples:

- batch
- platform
- cohort
- donor
- study
- disease state

atlasmtl uses domain information in two ways:

- optionally during training, to encourage more stable representations under
  distribution shift
- during benchmarking, to report robustness by subgroup

### Relationship between the four

The project should be understood in this order:

1. `label` is the main task
2. `hierarchy` constrains whether the label path is coherent
3. `coordinate` supports label rescue and atlas-aligned interpretation
4. `domain` explains where the model may fail to generalize and how robustness
   should be measured

This ordering is important for both architecture and benchmarking. When there
is a tradeoff, atlasmtl should prioritize reliable label transfer first, then
use hierarchy, coordinate, and domain handling to strengthen that primary goal.

## Training flow

### Inputs

`build_model()` consumes:

- `adata.X` as the expression matrix
- `adata.obs[label_columns]` as the multi-level labels
- optional `adata.obsm[coord_targets[*]]` as coordinate regression targets
- optional `adata.obs[domain_key]` as domain/batch metadata

### Data preparation

Training first normalizes the input contract:

- `extract_matrix()` applies the configured input transform
  - `"binary"` is the default
  - `"float"` keeps numeric expression values
- upstream preprocessing should canonicalize gene IDs before this step
  - preferred internal namespace is versionless Ensembl ID
  - readable symbols should be preserved separately in `adata.var`
- label columns are encoded with one `LabelEncoder` per level
- coordinate targets are standardized and their statistics are stored in
  `coord_stats`
- optional validation splitting is handled inside `build_model()`

Recommended feature policy:

- canonicalize the whole gene namespace first
- derive a reference-defined HVG panel second
- train and predict against that panel by exact gene-order alignment

Whole-matrix training is still possible, but it is not the preferred formal
protocol because the current architecture uses a relatively simple MLP encoder
with exact feature matching and a default binary transform.

The dedicated preprocessing implementation now lives under:

- `atlasmtl/preprocess/`

This keeps namespace normalization and feature selection out of the core
training and prediction modules.

### Preprocessing contract boundary

The current architecture should be read as a strict two-stage contract:

1. preprocessing defines the biological feature space and matrix semantics
2. core training and prediction consume that contract without silently
   redefining it

That means:

- preprocessing owns:
  - gene namespace canonicalization
  - counts detection
  - `counts_layer` validation
  - HVG panel construction
  - feature-panel persistence
- `build_model()` and `predict()` own:
  - expression extraction from already-aligned matrices
  - model fitting and inference
  - confidence, KNN, hierarchy, and output writeback

This separation avoids a major class of hidden conflicts where model code would
otherwise guess gene namespace or matrix semantics at runtime.

### Model body

The trainable model is a shared MLP encoder with:

- one classification head per label level
- one coordinate head per requested coordinate target

The intended semantics of the coordinate heads are:

- `latent` head
  - preferred geometric support for KNN correction
- `umap` head
  - preferred support for visualization and downstream interpretation

They are auxiliary supervision targets, not independent primary endpoints.

Optional train-time additions:

- domain alignment penalty
  - enabled by `domain_loss_weight > 0`
  - current implementation is lightweight mean-matching
- topology-aware coordinate penalty
  - enabled by `topology_loss_weight > 0`
  - current implementation preserves local neighbor distances within a
    minibatch

### Calibration fit

When validation data is available and `calibration_method="temperature_scaling"`
is requested:

- atlasmtl fits one temperature per label head
- the fitted values are stored in `train_config["calibration"]`
- inference can later apply them with `apply_calibration`

### Artifact creation

The training result is wrapped in `TrainedModel`, which stores:

- model weights
- label encoders
- train gene order
- coordinate target names and scaling statistics
- reference coordinates and labels for KNN
- train-time configuration and runtime summaries

Saving writes:

- `model.pth`
- `model_metadata.pkl`
- `model_feature_panel.json` when preprocessing captures a reference-derived panel
- `model_reference.pkl` or a custom external reference path
- `model_manifest.json`

The manifest is the stable automation entry point and records relative paths,
storage mode, input transform, and optional checksums.

## Prediction flow

### Inputs

`predict()` consumes:

- a `TrainedModel`
- a query `AnnData`
- optional KNN, calibration, open-set, and hierarchy controls

### Inference path

Prediction runs in this order:

1. align the query genes to the training gene order
2. extract the query matrix using the stored or overridden input transform
3. run the neural model in batches
4. unscale predicted coordinates back into `X_pred_*`
5. resolve the KNN space
   - prefer latent coordinates when both query and reference latent space are
     available
   - otherwise use UMAP
6. optionally apply post-hoc temperature scaling to logits
7. compute probabilities, max-prob confidence, and top1-top2 margin per label
   head
8. optionally apply KNN correction
   - `off`
   - `low_conf_only`
   - `all`
9. optionally apply open-set Unknown forcing
10. optionally enforce parent-child hierarchy consistency
11. return a `PredictionResult`

### Confidence and Unknown policy

The current prediction pipeline combines several signals:

- MTL confidence from softmax probabilities
- MTL margin from the top-1 vs top-2 probability gap
- optional KNN vote fraction
- optional open-set score

Closed-loop Unknown assignment currently uses:

- low MTL confidence
- whether KNN was used

## Benchmark architecture

### Current data-flow

The current benchmark path is:

1. dataset manifest is loaded by `benchmark/pipelines/run_benchmark.py`
2. optional atlasmtl preprocessing is applied once at the dataset level
3. preprocessed reference/query `.h5ad` files are written to the run folder
4. atlasmtl and comparators run against those concrete files
5. method results are merged into `metrics.json`, `summary.csv`, and
   `run_manifest.json`

### Comparator matrix semantics

Comparator wrappers are not all equivalent in how they consume expression
matrices:

- `scanvi`
  - requires a valid counts layer
- `singler`, `symphony`, `azimuth`
  - usually start from a configured raw-count layer and perform
    method-specific normalization inside the wrapper
- `celltypist`
  - primarily consumes the query `X` matrix through the Python API
- `reference_knn`
  - consumes `X` through atlasmtl's `extract_matrix(...)`

This is an intentional benchmark-layer distinction, not a core atlasmtl model
conflict. Formal interpretation should therefore compare methods on shared
label outputs, while recording matrix semantics in result metadata.

### Current architectural risks and resolved conflicts

Resolved in the current implementation:

- preprocessing and benchmark manifests now share explicit `input_matrix_type`
  and `counts_layer` metadata
- `seurat_v3` HVG selection is explicitly tied to `layer="counts"`
- comparator wrappers now resolve count-layer defaults from the benchmark
  manifest instead of each wrapper hardcoding `"counts"`

Remaining intentional boundaries:

- atlasmtl core still primarily reads `adata.X` during model training and
  prediction after preprocessing has already aligned the matrix
- external comparators do not share one universal matrix representation, so
  wrapper-level normalization remains method-specific
- whether KNN vote confidence stayed below `knn_conf_low`
- optional open-set Unknown mask

Hierarchy enforcement is an additional post-processing step and may replace
inconsistent child predictions with `Unknown`.

This ordering reflects the intended role of coordinates in atlasmtl:

- first provide a stable geometry for KNN rescue
- then provide atlas-aligned interpretation
- not replace the main label-centric decision path

## Export and writeback

`PredictionResult` supports three export modes:

- `minimal`
  - final labels only
- `standard`
  - final labels plus confidence, margin, and Unknown flags
- `full`
  - all debug columns, including raw labels, KNN labels, KNN vote fraction, and
    low-confidence flags

Writeback targets:

- `obs["pred_<level>"]`
- `obs["conf_<level>"]`
- `obs["margin_<level>"]`
- `obs["is_unknown_<level>"]`
- optional debug fields in full mode
- optional `obsm["X_pred_latent"]`
- optional `obsm["X_pred_umap"]`
- `uns["atlasmtl"]`

## Metadata and runtime summaries

### Train-time metadata

`TrainedModel.train_config` is the main train-time metadata snapshot. It may
include:

- core hyperparameters
- device and thread settings
- coordinate enablement
- train/validation sizes
- runtime summary
- resource summary
- calibration payload
- domain/topology settings
- preset selection

### Prediction metadata

`PredictionResult.metadata` and `uns["atlasmtl"]` may include:

- atlasmtl version
- confidence thresholds
- KNN thresholds and selected KNN space
- input transform
- calibration metadata and per-head temperatures
- prediction runtime summary
- KNN mode selections
- open-set settings and rates
- hierarchy enforcement settings and inconsistency rates
- the model `train_config`

## Benchmark architecture

The current benchmark path is intentionally minimal:

- one dataset manifest drives the run
- atlasmtl is currently the only implemented method wrapper
- the runner can either:
  - load a supplied atlasmtl artifact
  - or train one from the manifest config
- outputs are written as:
  - `metrics.json`
  - `summary.csv`
  - optional `summary_by_domain.csv`
  - `run_manifest.json`

Supported benchmark metric groups today:

- classification
- abstention/coverage
- calibration
- runtime/resource summaries
- artifact sizes
- optional coordinate RMSE and trustworthiness

## Current roadmap status

### Implemented in architecture

- post-hoc temperature scaling
- open-set scoring
- lightweight domain alignment
- optional hierarchy enforcement
- KNN voting variants, prototype reference mode, and approximate ANN hook
- topology-aware train-time coordinate regularization
- artifact checksums and run manifests

### Not yet closed in protocol or benchmarking

- hierarchy-aware benchmark metrics
- KNN variant benchmark matrix
- topology metrics beyond RMSE and trustworthiness
- domain-shift split definitions and failure-analysis reporting
- standardized split/seed traceability across all run manifests
- query-time adaptation

That gap is intentional: the code exposes the controls, but the benchmark and
protocol layer still needs to be hardened before those features should be
treated as fully complete roadmap items.
