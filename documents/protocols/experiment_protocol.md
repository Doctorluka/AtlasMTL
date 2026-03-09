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
- `seurat_anchor_transfer`

The benchmark reporting layer should be interpreted as two synchronized tables:

- performance tables
  - accuracy, macro-F1, calibration, coverage, and related benchmark metrics
- protocol tables
  - backend, target label, matrix sources, counts layer, normalization mode,
    and feature-alignment policy

Both are needed for paper-ready reporting.

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

## Benchmark ablation discussion

For the current ablation priorities and paper-facing resource reporting
requirements (including binary-vs-non-binary resource comparisons and CPU/GPU
variant handling), see:

- `documents/design/benchmark_ablation_discussion.md`
- `documents/experiments/2026-03-01_real_mapping_benchmark/ablation_optimization/`

Current locked ablation rules for AtlasMTL:

- source binary/float comparisons from the same `layers["counts"]` matrix
- define `float` as raw counts cast to `float32`, not `log1p`
- use `whole`, `hvg3000`, and `hvg6000` as the current feature-space grid
- require a benchmark-entry CUDA gate before admitting GPU variants

For follow-up AtlasMTL tuning, the selection target is not the single highest
accuracy point. The preferred outcome is an operational tradeoff between label
quality and resource cost:

- keep `anno_lv4` accuracy and macro-F1 as the primary quality targets
- treat training time, prediction time, peak RSS, and peak GPU memory as
  required co-primary resource constraints
- choose the lowest-resource configuration among candidates that remain close
  to the best observed quality, instead of hard-coding the numerically best
  accuracy run as the universal default

CellTypist implementation note:

- the benchmark repo still contains multiple CellTypist training paths:
  - `formal_native`
  - `formal_with_compat_shim`
  - `wrapped_logreg`
- formal third-wave reports must record the active backend path explicitly in
  machine-readable outputs and markdown summaries
- headline formal tables must not silently mix `wrapped_logreg` outputs with
  native CellTypist outputs
- do not compare wrapped-logistic CellTypist build times against historical
  PH-Map-style `celltypist.train(...)` figures without labeling the path
  difference explicitly
- detailed note:
  - `documents/protocols/celltypist_comparator_gap_note_2026-03-04.md`

Third-wave fairness note:

- stricter runtime fairness controls are defined in
  `documents/protocols/third_wave_fairness_protocol.md`
- this includes explicit thread-policy and degraded-runtime labeling rules
  (for example `joblib` serial fallback under restricted execution)

Current interpretation from the completed ablation round:

- `hvg6000 + binary + phmap` is the strongest current default candidate
- `whole` remains the stability baseline and must stay in future grids
- future HVG selection should be a local search around the current optimum,
  not an unrestricted sweep

Formal third-wave scaling note:

- the formal scaling contract is defined in
  `documents/protocols/formal_third_wave_scaling_protocol.md`
- build scaling uses a dedicated `build_eval_fixed_10k`
- predict scaling reuses the already-trained `100k` build artifact
- the build-scaling `10k` query and predict-scaling `10k` query must be
  different heldout subsets
- `Vento` is treated as a supplementary reduced-ceiling dataset in this formal
  round

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
- `input_matrix_type`
  - explicit input matrix declaration such as `infer`, `counts`, or `lognorm`
- `counts_layer`
  - raw-count layer name used by preprocessing, default recommendation:
    `counts`
- `gene_id_table`
  - optional path to a conversion table used during preprocessing
- `feature_space`
  - recommended values are `hvg` or `whole`
- `hvg_config`
  - optional HVG selection metadata such as method and target gene count

## Scenario classes and manifest naming

Until the runner grows a dedicated scenario-type field, formal runs should be
classified in documentation and file naming as one of these two classes:

- `reference_heldout`
  - formal quantitative benchmark using accepted truth labels from a reference
    dataset split into train/validation/test pools
- `external_query_validation`
  - deployment-style transfer into a query dataset that is not treated as
    formal benchmark truth by default

Recommended manifest file naming:

- `documents/experiments/<dossier>/manifests/<scenario_class>/<dataset_id>__<target_label>__<split_name>.yaml`

Examples:

- `documents/experiments/2026-03-01_real_mapping_benchmark/manifests/reference_heldout/PHMap_Lung_Full_v43_light__anno_lv4__group_split_v1.yaml`
- `documents/experiments/2026-03-01_real_mapping_benchmark/manifests/external_query_validation/HLCA_Core__ann_level_5__gse302339_marker_review_v1.yaml`

Recommended `split_name` patterns:

- `group_split_v1`
- `group_split_v1_train10k_test5k`
- `marker_review_v1`
- `labelset_shift_v1`

Use `split_description` to spell out:

- scenario class
- split field
- train/validation/test construction rule
- target label level
- whether the run is quantitative benchmark or visualization-first validation

## Reference-specific experiment organization

For the next benchmark wave, organize work **per reference dataset**, not as a
single monolithic experiment bundle.

Recommended rule:

- one reference dataset = one experiment dossier
- each dossier keeps its own:
  - manifests
  - scripts
  - notes
  - results summaries
  - output-root convention

This keeps dataset-specific differences explicit:

- sample-group column names differ
- label depth differs
- count contracts differ
- feasible scaling ceilings differ

Fairness rule:

- comparisons must still be fair **within** each scenario
- dataset-specific tuning is allowed only when:
  - it is required by the dataset contract or scale
  - it is applied consistently across compared methods where relevant
  - it is recorded in the protocol table and scenario note

Examples of acceptable dataset-specific adjustments:

- choosing `sample` vs `donor_id` vs `orig.ident` as the split/group field
- reducing the maximum training-size ceiling for a smaller reference atlas
- selecting the valid target label level for that dataset
- declaring `adata.X` counts vs `layers["counts"]` explicitly

Examples that require stronger justification:

- changing the optimization budget for only one comparator
- changing the evaluation target level mid-comparison
- changing the truth pool or split rule for one method but not others

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

## Output directory and run naming convention

Keep large runtime outputs under the user's private `~/tmp/` workspace.

Recommended output directory pattern:

- `~/tmp/atlasmtl_benchmarks/<date>/<scenario_class>/<dataset_id>/<split_name>/<method_set>/`

Recommended repo-side dossier pattern:

- `documents/experiments/<date>_<reference_id>_benchmark/`

Within that dossier, keep:

- `manifests/reference_heldout/`
- `manifests/external_query_validation/`
- `plan/`
- `notes/`
- `scripts/`
- `results_summary/`

Review each pilot manifest before execution with:

- `documents/protocols/pilot_benchmark_review_checklist.md`

Recommended `run_id` composition for notes, tables, and summaries:

- `<date>__<dataset_id>__<scenario_class>__<target_label>__<split_name>__seed<seed>`

## Second-wave scale-out defaults

For the current post-smoke expansion round, use these defaults unless a
dataset-specific ceiling forces an exception:

- preferred reference build target: `100k`
- preferred heldout targets: `10k` plus nested `5k`
- continue group-aware split construction by sample-like group
- continue per-reference dossier organization
- treat `Vento` as a reduced-ceiling case with `50k` build target by default

This round still targets **workflow expansion and reproducibility**, not final
paper-grade ranking.

Operational rule:

- if Seurat reference integration is unstable at smoke/scale-out size,
  `seurat_anchor_transfer` may fall back to `single_reference_pca`
- if `MapQuery`-style projection is unstable, it may fall back to
  `TransferData`
- the actual backend used must remain explicit in result metadata and reports

Minimum files that should be present in each output directory:

- `metrics.json`
- `summary.csv`
- `run_manifest.json`
- `benchmark_report.md` when report export is enabled
- `paper_tables/` when table export is enabled

When a scenario is part of a scaling study, append size metadata to
`split_name` instead of inventing an incompatible ad hoc directory layout.
Example:

- `group_split_v1_train10k_test5k`

## Local large dataset registry (non-versioned data)

Large datasets used for paper-grade evaluation are typically stored outside the
repo (e.g. under `/home/data/...`). To reduce future “path drift” and contract
mismatches, record:

- dataset path
- observed gene namespace (`symbol` vs `ensembl`)
- raw-count location (`layers["counts"]` vs `raw.X`)
- available label columns (single-level vs multi-level)
- available `obsm/obsp` geometry assets (when KNN/coordinates are evaluated)

The current local dataset inventory is tracked in:

- `data_registry/datasets.md`

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
- `input_matrix_type`
  - whether the incoming `adata.X` should be interpreted as `infer`,
    `counts`, or `lognorm`
- `counts_layer`
  - the layer name containing raw counts, defaulting to `counts`
- `gene_id_table`
  - conversion table or resource identifier when remapping is applied
- duplicate-handling policy
- unmapped-gene policy
- whether `adata.X` was count-like at load time
- the raw-count layer name, defaulting to `counts`

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

- validate whether `adata.X` is count-like before any feature selection
- if `adata.X` is count-like, persist it to `adata.layers["counts"]`
- if `adata.X` is not count-like, require `adata.layers["counts"]` and fail otherwise
- `feature_space="hvg"`
- `n_top_genes=3000`
- `hvg_method="seurat_v3"` with `layer="counts"`
- do not generate log-normalized matrices inside atlasmtl core preprocessing by default
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
| `seurat_anchor_transfer` | reference mapping | `Seurat v5` R libs | one target label level | label mapping quality | Seurat anchor-transfer plus `MapQuery`; follow benchmark `HVG 3000` policy |

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
- Do not relabel a generic Seurat anchor-transfer run as `Azimuth`. Report this
  comparator explicitly as `seurat_anchor_transfer`.
- Do not claim full multi-level fairness for comparators that only output a
  single label level.

## Benchmark environments

- Python env:
  `/home/data/fhz/.local/share/mamba/envs/atlasmtl-env`
- Recommended `NUMBA_CACHE_DIR`:
  `/tmp/numba_cache`
- `Seurat v5` R library:
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
- average RSS when available
- average GPU memory when available
- average CPU utilization / core-equivalent usage when available
- CPU thread count / effective CPU core usage when available
- device used (`cpu` or `cuda`)
- artifact sizes in MB

These are supporting engineering metrics.

For formal benchmark reporting, the preferred resource table should include at
least:

- train elapsed seconds
- predict elapsed seconds
- train peak RSS
- predict peak RSS
- train peak GPU memory
- predict peak GPU memory
- average memory where the runner can provide it
- average GPU memory where the runner can provide it
- average CPU utilization or core-equivalent usage where the runner can provide it
- CPU thread/core count used
- device used

When atlasmtl is benchmarked on both CPU and GPU, treat them as separate
benchmark variants and label them explicitly in manifests, result tables, and
paper figures.

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

- Seurat anchor transfer / `MapQuery` reference mapping
- Symphony
- scANVI
- SingleR
- CellTypist

These methods belong to the same `sc -> sc reference mapping` task family.

### Locked scANVI default for formal runs

Based on the completed pre-formal parameter confirmation experiment on
`2026-03-06`, the default `scanvi` setting for subsequent formal experiments is
locked as:

- `scvi_max_epochs=25`
- `scanvi_max_epochs=25`
- `query_max_epochs=20`
- `n_latent=20`
- `batch_size=256`
- `datasplitter_num_workers=0`

Execution rule:

- in this project phase, `scanvi` is evaluated in `GPU` mode only for formal
  benchmark runs.
- if a future run must deviate from this lock, the deviation must be documented
  explicitly as an ablation or exception note.

Evidence dossier:

- `documents/experiments/2026-03-06_scanvi_param_lock_benchmark/results_summary/experiment_report_2026-03-06_scanvi_param_lock.md`
- `documents/experiments/2026-03-06_scanvi_param_lock_benchmark/results_summary/param_lock_decision_2026-03-06.md`

### Locked AtlasMTL defaults for formal runs

Based on the completed pre-formal parameter confirmation experiment on
`2026-03-07`, the default `atlasmtl` settings for subsequent formal experiments
are locked as:

- shared fixed settings:
  - `num_threads=8`
  - `max_epochs=50`
  - `val_fraction=0.1`
  - `early_stopping_patience=5`
  - `input_transform=binary`
  - `reference_storage=external`
- CPU default:
  - `learning_rate=3e-4`
  - `hidden_sizes=[256,128]`
  - `batch_size=128`
- GPU default:
  - `learning_rate=1e-3`
  - `hidden_sizes=[1024,512]`
  - `batch_size=512`

Execution rule:

- in formal reporting, `atlasmtl` CPU and `atlasmtl` GPU should be treated as
  separate runtime/resource variants.
- if a future formal run deviates from these defaults, the deviation must be
  documented explicitly as an ablation or exception note.

Evidence dossier:

- `documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/results_summary/experiment_report_2026-03-07_atlasmtl_param_lock.md`
- `documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/results_summary/atlasmtl_lock_decision_2026-03-07.md`

Deferred to later ablation round:

- `domain_loss_weight`
- `topology_loss_weight`
- `calibration_method`

Formal tables should label CPU and GPU atlasmtl as separate variants when
reporting runtime or throughput.

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
- path handling rule:
  - repo-tracked manifests should prefer atlasmtl-root-relative paths
  - the benchmark runner resolves a relative path against the manifest
    directory first, then against the repository root
  - run-generated manifests may record absolute paths for that concrete run
