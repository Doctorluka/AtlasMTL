# atlasmtl Research Positioning

## One-sentence positioning

`atlasmtl` is a reference mapping method for multi-level cell annotation whose
primary goal is accurate, reliable, and abstention-aware label transfer, rather
than building a general-purpose integrated latent embedding benchmark winner.

## Primary research question

The primary question for atlasmtl is:

- can a reference mapping workflow produce cell labels that are accurate,
  reliable, hierarchically consistent, and operationally useful under realistic
  uncertainty?

In this framing, the core outputs are:

- final labels
- confidence and margin signals
- Unknown / abstention behavior
- KNN rescue behavior on difficult cells
- hierarchy consistency

## What atlasmtl is not primarily trying to prove

atlasmtl is not currently positioned around this stronger claim:

- the latent representation learned by atlasmtl is itself a state-of-the-art
  integrated embedding that should be evaluated as a direct competitor to
  scIB-style integration methods

That stronger positioning would require a different benchmark emphasis,
different baselines, and a larger methodological burden around batch correction
and biological conservation.

## Why this is the right positioning

### 1. It matches the 2016-2026 literature trend

The reference review in
`documents/reference/methodology_of_reference_mapping_2016_2026.pdf` points to
a task-first framing:

- `sc -> sc reference mapping` should be benchmarked as label transfer /
  annotation
- `sc -> ST deconvolution`, localization, and gene enhancement are separate
  tasks with different output semantics and different metrics
- for atlas-style mapping, the stable evaluation focus is label quality,
  abstention, calibration, and robustness

The literature does not require every mapping method to prove that its latent
space is the best possible integrated embedding.

### 2. It matches atlasmtl's implemented strengths

atlasmtl already has concrete machinery for:

- multi-level labels
- confidence and margin outputs
- Unknown / abstention
- KNN rescue and KNN audit columns
- hierarchy enforcement
- coordinate-aware correction support
- run manifests and artifact traceability

These features support a strong reliability-and-annotation story today.

### 3. It avoids an unnecessary benchmark burden

If atlasmtl were positioned primarily as an integration method, it would need
to prove more than it currently implements:

- competitive scIB-style batch mixing and biological conservation
- stronger alignment objectives than the current lightweight domain penalty
- fair head-to-head comparison against methods explicitly optimized for
  integrated latent spaces, such as scANVI, scArches, Symphony, and related
  frameworks

That is a valid future direction, but it is not the right primary claim for the
current project stage.

## Benchmark implications

### Primary benchmark track

atlasmtl benchmarking should center on:

- `sc -> sc reference mapping`
- multi-level cell annotation
- reliability under uncertainty
- domain robustness when labels are transferred across batches, cohorts, or
  studies

### Primary benchmark metrics

The main benchmark tables should prioritize:

- `accuracy`
- `macro_f1`
- `balanced_accuracy`
- `coverage`
- `reject_rate`
- `covered_accuracy`
- `risk`
- `ece`
- `brier`
- `aurc`

atlasmtl-specific behavior metrics should also be first-class:

- `unknown_rate`
- `knn_coverage`
- `knn_rescue_rate`
- `knn_harm_rate`
- hierarchy path consistency

### Secondary / supporting analyses

Coordinate and embedding-oriented diagnostics are still useful, but they are
supporting evidence rather than the primary success criterion:

- `rmse`
- `trustworthiness`
- `continuity`
- `neighbor_overlap`

These metrics help explain whether coordinate-aware correction behaves
sensibly. They should not be treated as the main research claim.

## Comparator strategy

The first benchmark comparator set should stay within the same task family:

- Seurat / Azimuth-style anchor transfer
- Symphony
- scANVI
- SingleR
- CellTypist

These are appropriate because they all operate in the `sc -> sc reference
mapping` space. Methods specialized for deconvolution, localization, or gene
imputation should not be used as primary comparators for the core atlasmtl
claim.

## Out-of-scope benchmark tasks for the current project stage

The following should not be mixed into the main atlasmtl benchmark:

- `sc -> ST deconvolution`
- single-cell localization into real spatial coordinates
- spatial gene imputation / enhancement
- foundation-model-style zero-shot representation benchmarking

Those may become future extensions, but they should not define the current
methodology.

## Practical guidance for future development

When there is a tradeoff between:

- improving label reliability, abstention quality, KNN rescue quality, and
  traceability
- or improving generic latent integration metrics

the first category takes priority unless the project explicitly changes its
research positioning.

Likewise, benchmark additions should first answer:

- is atlasmtl producing better and more reliable labels?

before asking:

- is atlasmtl also a better integrated embedding model?

## Exit condition for changing this positioning

The project should only upgrade to a stronger integration-centric positioning if
all of the following become true:

- domain-shift protocol and comparator benchmarking are mature
- atlasmtl includes stronger alignment objectives than the current lightweight
  penalty
- scIB-style integration metrics are intentionally adopted
- the paper claims are rewritten around integrated representation quality, not
  just reliable annotation
