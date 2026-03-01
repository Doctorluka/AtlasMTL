# atlasmtl Overall Summary

## What atlasmtl is

`atlasmtl` is a single-cell `sc -> sc` reference mapping framework centered on
reliable multi-level label transfer. Its primary goal is not to produce the
best generic integrated embedding, but to transfer cell labels accurately,
calibrate uncertainty, support abstention when evidence is weak, and preserve
traceability from model artifacts to benchmark outputs.

In practical terms, atlasmtl is designed to answer this question:

- given a labeled reference atlas and an unlabeled query dataset, can we assign
  biologically meaningful labels in a way that is accurate, reliable,
  uncertainty-aware, and operationally usable?

That positioning determines the framework design, the metric focus, and the
benchmark protocol.

## Core purpose

The framework exists to solve a common single-cell atlas problem:

- reference atlases provide curated label systems
- new query datasets need to be mapped into those systems
- the mapping must handle uncertainty, distribution shift, and label hierarchy
- the outputs must be easy to write back to `AnnData`, benchmark, and audit

atlasmtl therefore aims to provide:

- multi-level annotation
- explicit confidence and margin outputs
- optional Unknown / abstention behavior
- optional KNN rescue for difficult cells
- hierarchy-aware consistency checks
- artifact and run-level traceability

## Framework design

### Design principle

atlasmtl is structured as a thin public API on top of a modular training,
prediction, mapping, I/O, and serialization stack.

This keeps the user contract simple:

- `build_model()`
- `predict()`
- `TrainedModel`
- `PredictionResult`

while allowing the internals to evolve independently.

### Main modules

- `atlasmtl/core/`
  - training, inference, evaluation, runtime summaries, shared types
- `atlasmtl/mapping/`
  - calibration, KNN logic, open-set scoring, hierarchy enforcement
- `atlasmtl/models/`
  - serialization, manifests, checksums, presets, reference storage
- `atlasmtl/io/`
  - AnnData writeback and export helpers
- `atlasmtl/utils/`
  - progress and runtime helpers
- `benchmark/`
  - comparator wrappers, benchmark runner, protocol-oriented outputs

This separation reflects the project’s priorities: stable user entrypoints,
predictable artifact layout, and benchmark reproducibility.

## Gene identifier and feature policy

atlasmtl currently performs exact gene-name alignment using `adata.var_names`.
That means the project is only as robust as its upstream identifier
canonicalization.

The intended policy is:

- internal computation key
  - versionless Ensembl ID
- readable annotation field
  - `adata.var["gene_symbol"]`

This split is deliberate:

- Ensembl is more stable for cross-dataset exact matching
- symbols are better for interpretation and figures

Because atlasmtl still defaults to a binary input transform, gene-ID mistakes
are especially expensive: a mismatch is not just a softer quantitative change,
it can zero out the entire feature column after alignment.

For that reason, formal runs should require preprocessing metadata describing:

- whether the original `var_names` were `symbol` or `ensembl`
- which species the data belongs to
- what mapping table or resource was used
- how duplicate or unmapped genes were handled

atlasmtl now bundles a baseline cross-species mapping resource at:

- `atlasmtl/resources/gene_id/biomart_human_mouse_rat.tsv.gz`

This file is suitable as a packaged preprocessing input, but it should still
be accompanied by explicit run metadata describing namespace, species, and
mapping decisions.

### Whole matrix vs HVG

The preferred formal training policy is:

- canonicalize the full gene namespace first
- select a reference-derived HVG panel second
- train and benchmark on that panel

Whole-matrix training remains useful for ablations and some resource-rich
settings, but it should not be the default protocol. The current atlasmtl
stack uses an MLP encoder plus exact feature alignment, so whole-matrix runs
usually cost more in memory and runtime than they return in label-transfer
quality.

The recommended default for future preprocessing utilities is:

- `feature_space="hvg"`
- `n_top_genes=3000`

Whole-matrix mode should require an explicit user choice.

That preprocessing layer is now implemented as a dedicated package under:

- `atlasmtl/preprocess/`

## Main algorithmic structure

### 1. Shared multi-task encoder

The model uses a shared encoder to learn a common representation from the
reference expression matrix. On top of this shared representation, atlasmtl
attaches:

- one classification head per label level
- optional coordinate regression heads

This is why the method is called “atlasmtl”: the central inductive bias is
multi-task learning across annotation levels and optional atlas coordinates.

### 2. Multi-level label prediction

The primary supervised task is label transfer across one or more ordered label
columns, for example:

- coarse cell class
- intermediate cell subtype
- fine cell state

Each label level is trained with its own head while sharing the same encoder.
This lets the framework reuse common signal while preserving level-specific
decision boundaries.

### 3. Optional coordinate regression

atlasmtl can also regress query cells into atlas-aligned coordinate spaces,
such as:

- latent coordinates
- UMAP coordinates

These predicted coordinates are not the primary endpoint. They are supporting
structure used to:

- provide a geometric space for KNN correction
- interpret the mapping geometry
- measure whether mapped cells land in a sensible atlas neighborhood

The intended internal priority is:

- `latent` coordinates
  - primary correction space when available
- `umap` coordinates
  - mainly a visualization and interpretation space

In other words, coordinate regression is not a fully independent co-equal task.
It is an auxiliary atlas-alignment module whose first job is to support
label-transfer reliability, especially KNN rescue.

### 4. Confidence and reliability layers

atlasmtl explicitly models prediction reliability through several mechanisms:

- max-probability confidence
- top-1 vs top-2 margin
- optional post-hoc calibration
- optional open-set scoring
- optional Unknown forcing

These layers make the framework operationally safer than a plain forced-label
transfer pipeline.

### 5. KNN rescue and audit

atlasmtl supports KNN-based correction with configurable behavior:

- correction mode
- vote mode
- reference representation mode
- index mode

The point of this layer is not to overwrite all model predictions. It is to
help low-confidence or difficult cells using reference neighborhood context,
while exposing audit signals such as:

- whether KNN was used
- whether it changed the label
- whether it helped or harmed

### 6. Hierarchy enforcement

When label levels form a parent-child system, atlasmtl can enforce hierarchy
rules at prediction time. This treats hierarchy as a structural validity layer
over the predicted labels.

For example:

- a coarse immune label should not coexist with a child label outside the
  immune branch

Invalid child predictions can be forced to `Unknown`, and inconsistency rates
can be recorded for benchmark analysis.

### 7. Domain robustness

atlasmtl can optionally read a `domain_key`, such as:

- batch
- platform
- donor
- study

It uses this information in two ways:

- optional train-time domain alignment penalty
- benchmark-time grouped robustness reporting

This makes domain information a robustness tool, not a primary prediction
target.

### 8. Topology-aware coordinate regularization

atlasmtl includes optional topology-aware coordinate loss terms. These attempt
to preserve local neighborhood structure inside minibatches for chosen
coordinate targets.

This is intended to improve the geometric quality of coordinate outputs and the
stability of coordinate-aware correction. It is a supporting mechanism rather
than the central method claim.

## Implementation pathway

### Training pathway

At training time, atlasmtl:

1. reads the reference `AnnData`
2. extracts the input matrix with the configured transform
3. encodes one or more label columns
4. optionally reads coordinate targets and domain labels
5. trains the shared encoder plus task heads
6. optionally fits calibration parameters on validation data
7. packages weights, encoders, reference information, and config into
   `TrainedModel`

### Prediction pathway

At prediction time, atlasmtl:

1. aligns query genes to the training gene order
2. runs the neural model in batches
3. computes label probabilities and margins
4. optionally writes predicted coordinates
5. optionally applies calibration
6. optionally applies KNN correction
7. optionally applies open-set Unknown logic
8. optionally enforces hierarchy constraints
9. returns a `PredictionResult` and supports AnnData writeback

### Artifact pathway

atlasmtl places strong emphasis on reproducibility. The preferred artifact
bundle is:

- `model.pth`
- `model_metadata.pkl`
- `model_feature_panel.json` when preprocessing defines a feature panel
- `model_reference.pkl`
- `model_manifest.json`

The manifest is the stable loading entrypoint and can include checksums. In
addition, train/predict/benchmark runs can write run manifests for downstream
traceability.

## Benchmark design

### Benchmark philosophy

atlasmtl is benchmarked first as a label-transfer method, not as an integrated
embedding benchmark winner. This means the main comparison question is:

- which method transfers the target label most accurately and reliably?

The main metrics are therefore:

- `accuracy`
- `macro_f1`
- `balanced_accuracy`
- `coverage`
- `reject_rate`
- `ece`
- `brier`
- `aurc`

atlasmtl-specific secondary analyses include:

- Unknown rate
- KNN rescue / harm
- hierarchy consistency
- coordinate/topology diagnostics

### Current comparator set

The current runnable benchmark methods are:

- `atlasmtl`
- `reference_knn`
- `celltypist`
- `scanvi`
- `singler`
- `symphony`
- `azimuth`

These comparators were chosen because they belong to the same broad
`sc -> sc` reference mapping / annotation family.

### Fairness boundary

Not all comparators support the full atlasmtl feature set. Most external
comparators are still evaluated as single-level methods. Therefore the fairest
current comparison is:

- same reference/query split
- same target label column
- same held-out truth labels
- same primary scoring metrics

atlasmtl’s hierarchy, KNN rescue, open-set behavior, and coordinate-aware
machinery should be interpreted as added method value, not as a requirement
imposed on every comparator.

For paper framing, coordinate outputs should therefore be described as:

- auxiliary atlas-alignment outputs
- primary support for KNN correction
- secondary support for visualization and biological interpretation

They should not be framed as a separate main prediction task on equal footing
with the label-transfer objective.

## Environments and runtime stack

### Python

Primary development and benchmark environment:

- `/home/data/fhz/.local/share/mamba/envs/atlasmtl-env`

Recommended runtime setting for `scanpy`-related commands:

- `NUMBA_CACHE_DIR=/tmp/numba_cache`

### R

Current comparator stack uses two R library locations:

- native `Azimuth` / `Seurat v5`:
  `/home/data/fhz/seurat_v5`
- repo-local comparator R library:
  `/home/data/fhz/project/phmap_package/atlasmtl/.r_libs`

These paths matter operationally and should be documented whenever benchmark
results are reported.

## Main uses

atlasmtl is useful when a project needs:

- reference-based cell annotation
- multi-level labels instead of a single flat class
- uncertainty-aware prediction outputs
- auditable KNN rescue behavior
- hierarchy-aware label validity
- reproducible model packaging and benchmark runs

It is particularly suitable for atlas-style workflows where the reference is
curated and stable, and where the user needs more than a forced single-label
assignment.

## Current strengths

- stable `AnnData in, AnnData out` contract
- multi-level label transfer
- calibration, open-set, and abstention support
- KNN rescue variants and auditability
- hierarchy enforcement
- manifest-based artifact loading and checksums
- benchmark runner with multiple published comparators

## Current limitations

- most external comparators are only single-level in the current benchmark
- domain-shift protocol is not fully standardized yet
- comparator matrix execution is not fully automated yet
- topology and hierarchy metrics are not yet uniformly comparable across all
  methods
- native `Azimuth` can require fallback on very small synthetic smoke datasets

## Practical interpretation

atlasmtl should currently be described as:

- a reliable multi-level `sc -> sc` reference mapping framework
- with explicit uncertainty handling
- with optional geometric and structural correction layers
- and with increasing benchmark maturity across multiple comparator families

That description is both methodologically accurate and aligned with the current
state of the repository.
