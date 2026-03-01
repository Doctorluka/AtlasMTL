# atlasmtl Experiment Protocol

## Scope

This document defines the current minimum experiment and benchmark protocol for
atlasmtl. It is designed to match the current project positioning:

- atlasmtl is benchmarked primarily as a reliable `sc -> sc reference mapping`
  method for multi-level label transfer
- atlasmtl is not benchmarked primarily as a general-purpose integrated
  embedding method

See `documents/design/research_positioning.md` for the full rationale.

## Benchmark entry point

The current benchmark entry point is:

`benchmark/pipelines/run_benchmark.py`

The runner currently supports atlasmtl plus these comparator methods:

- `reference_knn`
- `celltypist`
- `scanvi`
- `singler`
- `symphony`
- `azimuth`

## Benchmark philosophy

The main benchmark question is:

- are the transferred labels accurate, reliable, hierarchically coherent, and
  operationally useful under uncertainty?

The benchmark is not currently designed to answer this stronger question:

- is atlasmtl's latent space itself a state-of-the-art integrated embedding
  under scIB-style criteria?

Accordingly:

- label-quality, calibration, abstention, KNN rescue, and robustness metrics
  are primary
- coordinate and topology diagnostics are secondary
- batch correction and biological conservation metrics are not primary protocol
  requirements

## Dataset manifest schema

The benchmark runner expects a YAML mapping with these required keys:

- `reference_h5ad`
  - path to the reference AnnData file used for training or model validation
- `query_h5ad`
  - path to the query AnnData file used for prediction and scoring
- `label_columns`
  - ordered list of label columns present in both reference and query `obs`

Supported optional keys:

- `dataset_name`
  - human-readable dataset identifier
- `version`
  - dataset version string
- `coord_targets`
  - mapping from internal coordinate head names to reference `obsm` keys
- `query_coord_targets`
  - mapping from internal coordinate head names to query `obsm` keys for
    coordinate scoring
- `domain_key`
  - `query.obs` column used for grouped reporting
- `train`
  - training configuration block used when the runner trains atlasmtl itself
- `predict`
  - prediction configuration block passed to `predict()`
- `var_names_type`
  - declared input gene namespace such as `symbol` or `ensembl`
- `species`
  - species label such as `human`, `mouse`, or `rat`
- `gene_id_table`
  - optional path to a conversion table used during preprocessing
- `feature_space`
  - recommended values are `hvg` or `whole`
- `hvg_config`
  - optional HVG selection metadata such as method and target gene count

### Recommended optional protocol extensions

These keys are not enforced yet by the current runner but should be added to
dataset manifests for stronger traceability:

- `protocol_version`
- `random_seed`
- `split_name`
- `split_description`
- `reference_subset`
- `query_subset`

Those fields should be treated as advisory until the runner validates them
directly.

## Gene namespace and feature-space policy

atlasmtl currently aligns genes by exact `adata.var_names` matching. Because
the runtime does not yet include a synonym resolver, formal experiments should
standardize the gene namespace before model training and prediction.

### Canonical internal namespace

The recommended internal computation key is:

- versionless Ensembl ID

Recommended practice:

- convert input `var_names` to Ensembl stable IDs before atlasmtl alignment
- strip Ensembl version suffixes such as `.7`
- store readable symbols in `adata.var["gene_symbol"]`
- record the mapping resource used to convert from symbols or other IDs

Bundled baseline mapping resource:

- `atlasmtl/resources/gene_id/biomart_human_mouse_rat.tsv.gz`
  - current scope: human, mouse, rat
  - intended use: preprocessing support, not silent runtime remapping

### Required preprocessing metadata

Formal experiments should record:

- `var_names_type`
  - original namespace, for example `symbol` or `ensembl`
- `species`
  - for example `human`, `mouse`, or `rat`
- `gene_id_table`
  - conversion table or resource identifier when remapping is applied
- duplicate-handling policy
- unmapped-gene policy

### Feature-space recommendation

For formal atlasmtl runs, the preferred feature-space policy is:

- canonicalize genes first
- derive a reference-defined HVG panel second
- subset query data to the trained reference feature panel at inference time

This is preferred because atlasmtl currently uses:

- exact feature alignment
- an MLP-style encoder
- default binary input transform

Under this design, whole-matrix training usually increases memory and runtime
more than it improves the main label-transfer objective. Whole-matrix runs
should be treated as ablations or special cases rather than the default
protocol.

Default recommendation for future preprocessing utilities:

- `feature_space="hvg"`
- `n_top_genes=3000`
- whole matrix must be explicitly requested with `feature_space="whole"`

This preprocessing layer is now implemented as explicit public APIs:

- `preprocess_reference(...)`
- `preprocess_query(...)`

## Training config block

The `train` block may contain any subset of the currently wired benchmark
training parameters:

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
- `calibration_method`
- `calibration_max_iter`
- `calibration_lr`
- `reference_storage`
- `reference_path`

This block currently does not expose all Python API training controls. In
particular, the benchmark runner does not yet pass through the full domain,
topology, or preset surface.

## Prediction config block

The `predict` block may currently contain:

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

The runner still does not expose the full Python prediction surface, but it now
supports hierarchy rules and the main KNN variant controls needed for benchmark
ablations.

## Comparator benchmark protocol table

The table below defines the current fairness contract for comparator reporting.
The main goal is not to force every method into the full atlasmtl feature set,
but to compare them on the shared question: can they transfer a target label
accurately and reliably in `sc -> sc` reference mapping?

| Method | Family | Runtime env | Typical target | Primary comparison scope | Notes |
|---|---|---|---|---|---|
| `atlasmtl` | primary method | Python env `atlasmtl-env` | one or more label levels | full protocol | Supports hierarchy, KNN rescue, calibration, open-set, and coordinate diagnostics |
| `reference_knn` | local baseline | Python env `atlasmtl-env` | one target label level | label transfer baseline | Minimal non-parametric reference vote |
| `celltypist` | supervised annotator | Python env `atlasmtl-env` | one target label level | label accuracy + confidence | Evaluated as single-level comparator |
| `scanvi` | probabilistic generative model | Python env `atlasmtl-env` | one target label level | label accuracy + confidence | Evaluated as single-level comparator |
| `singler` | classical reference annotator | R bridge + `atlasmtl-env` helper | one target label level | label accuracy + abstention | Uses `pruned.labels` as default benchmark-facing output |
| `symphony` | reference mapping | R bridge + repo-local R libs | one target label level | label mapping quality | Mapping + KNN label transfer |
| `azimuth` | reference mapping | native `Azimuth` / `Seurat v5` R libs | one target label level | label mapping quality | Prefer `azimuth_native`; fallback backend is testing-only unless explicitly stated |

### Fairness rules

- Use the same reference, query, and held-out truth labels across all methods.
- Fix one shared `target_label_column` for external comparator comparisons.
- Use shared primary metrics:
  - `accuracy`
  - `macro_f1`
  - `balanced_accuracy`
  - `coverage`
  - `reject_rate`
  - `ece`
  - `brier`
  - `aurc`
- Treat hierarchy consistency, KNN rescue/harm, open-set behavior, and
  coordinate/topology diagnostics as atlasmtl-centered secondary analyses
  unless an external comparator exposes a truly comparable signal.
- Do not mix native `azimuth` results with fallback results in a formal main
  table. If fallback is used, record it in metadata and treat it as a smoke or
  engineering validation path.
- Do not claim full multi-level fairness for comparators that only output a
  single label level.

## Benchmark environments

- Python env:
  `/home/data/fhz/.local/share/mamba/envs/atlasmtl-env`
- Recommended `NUMBA_CACHE_DIR`:
  `/tmp/numba_cache`
- Native `Azimuth` / `Seurat v5` R library:
  `/home/data/fhz/seurat_v5`
- Repo-local comparator R library:
  `/home/data/fhz/project/phmap_package/atlasmtl/.r_libs`

For a fillable benchmark reporting skeleton, use:

- `documents/protocols/comparator_benchmark_result_template.md`

## Output files

The current benchmark runner writes these files under `--output-dir`.

### `metrics.json`

Top-level fields:

- `protocol_version`
- `dataset_manifest`
- `results`

Each atlasmtl result payload currently contains:

- `method`
- `dataset_name`
- `dataset_version`
- `label_columns`
- `metrics`
- `metrics_by_domain`
- `behavior_metrics`
- `behavior_metrics_by_domain`
- `hierarchy_metrics`
- `coordinate_metrics`
- `train_usage`
- `predict_usage`
- `artifact_sizes`
- `artifact_paths`
- `artifact_checksums`
- `model_source`
- `model_input_path`
- `train_config_used`
- `predict_config_used`
- `prediction_metadata`

### `summary.csv`

One row per `(method, dataset_name, level)` with flat metric columns.

### `summary_by_domain.csv`

Written only when:

- the dataset manifest provides `domain_key`
- the corresponding column exists in `query.obs`

One row per `(method, dataset_name, domain, level)`.

### `run_manifest.json`

Current fields:

- `schema_version`
- `dataset_manifest`
- `methods`
- `python`
- optional `artifact_paths`
- optional `artifact_checksums`

## Metric definitions

### Classification metrics

Computed per label level:

- `accuracy`
- `macro_f1`
- `balanced_accuracy`

These are primary metrics.

### Abstention and coverage metrics

Computed per label level:

- `coverage`
- `reject_rate`
- `n_total`
- `n_covered`
- `covered_accuracy`
- `risk`

These are primary metrics.

### Prediction behavior metrics

Computed per label level:

- `unknown_rate`
- `knn_coverage`
- `knn_change_rate`
- `knn_change_rate_among_used`
- `knn_rescue_rate`
- `knn_rescue_rate_among_used`
- `knn_harm_rate`
- `knn_harm_rate_among_used`

These are primary atlasmtl behavior metrics because the method explicitly
claims abstention-aware and rescue-aware label transfer.

### Calibration metrics

Computed only when `conf_<level>` columns exist:

- `ece`
- `brier`
- `aurc`

These are primary metrics whenever confidence outputs are available.

### Runtime and artifact metrics

Reported through `train_usage`, `predict_usage`, and `artifact_sizes`:

- elapsed seconds
- throughput
- peak RSS
- peak GPU memory when available
- artifact sizes in MB

These are supporting engineering metrics.

### Coordinate metrics

Computed only when:

- the runner has predicted coordinates for a named head
- the dataset manifest provides the matching query target

Current coordinate metrics:

- `*_rmse`
- `*_trustworthiness`
- `*_continuity`
- `*_neighbor_overlap`

These are supporting diagnostics. They help interpret coordinate-aware mapping
behavior, but they are not the main success criteria for atlasmtl.

### Hierarchy metrics

Computed when `predict.hierarchy_rules` is provided:

- `full_path_accuracy`
- `full_path_coverage`
- `full_path_covered_accuracy`
- per-edge `path_consistency_rate`

These are primary metrics whenever hierarchy rules are supplied.

## Comparator guidance

The intended first-line comparator family is:

- native `Azimuth` / Seurat reference mapping
- Symphony
- scANVI
- SingleR
- CellTypist

These methods belong to the same `sc -> sc reference mapping` task family.

The following are not primary comparators for the current atlasmtl claim:

- deconvolution-specific methods such as cell2location, CARD, stereoscope, or
  related spot-composition models
- localization-specific methods such as Tangram, CytoSPACE, SpaOTsc, or
  CellTrek
- gene-enhancement methods such as SpaGE or gimVI

Those methods answer different task definitions and should not be mixed into the
main benchmark.

## Hardware and environment recording

Each benchmark run should record:

- Python executable
- dataset manifest path
- method list
- train/predict runtime summaries from atlasmtl metadata
- active comparator runtime environment paths when applicable, especially:
  - `/home/data/fhz/.local/share/mamba/envs/atlasmtl-env`
  - `/home/data/fhz/seurat_v5`
  - `/home/data/fhz/project/phmap_package/atlasmtl/.r_libs`

The broader experiment context should also be recorded outside the runner when
possible:

- exact package environment
- hardware model information
- `NUMBA_CACHE_DIR` if relevant for `scanpy`

## Randomness and split recording

Current enforced behavior:

- atlasmtl training can record `random_state` through the `train` block
- train runtime metadata is embedded in `train_config`

Current protocol expectation, even when not enforced by code:

- record the seed used for each benchmark run
- record whether the run uses a fixed artifact or trains within the runner
- record split definitions in the dataset manifest

Split recording is still weaker than the roadmap target and remains an active
gap.

## Domain-shift reporting

Current grouped reporting uses `evaluate_predictions_by_group()` over
`query.obs[domain_key]`.

This is useful for:

- per-batch performance summaries
- per-platform summaries
- per-cohort summaries

But this is not yet a full domain-shift protocol. The runner does not currently
standardize:

- in-domain vs cross-domain split definitions
- held-out cohort semantics
- interpretation of grouped Unknown/KNN rescue metrics as a formal shift test

Those remain roadmap work.

## Known gaps against the roadmap

The protocol is intentionally incomplete in these areas:

- no comparator baselines
- no comparator-aligned hierarchy metric block
- no full KNN variant benchmark matrix orchestration beyond a single predict config
- no full topology metric block beyond per-run coordinate summaries
- no enforced split schema
- no fully standardized seed and provenance schema across every run manifest
- no query-time adaptation protocol
- no scIB-style latent integration benchmark, because that is outside the
  current primary project positioning
