# AtlasMTL Methods Supplement Notes

This note records implementation-level details that are useful for supplement,
repository-linked methods, or reviewer response material, but are not required
to stay in the main Methods text.

## 1. Architecture-Level Implementation Details

- The repository-facing default model family is a lightweight MLP with hard
  parameter sharing.
- The current implementation default uses hidden sizes `[256, 128]` and
  `dropout_rate=0.3`.
- The shared encoder is followed by one linear head per label column.
- The framework also supports optional coordinate heads for extended
  configurations.

Use in paper:

- suitable for supplement or implementation details
- not necessary in the main Methods unless reviewers require exact default
  widths

## 2. Extended Objective Components

Beyond the primary weighted multi-task classification objective, the framework
can optionally include:

- coordinate reconstruction / regression losses
- lightweight domain-alignment penalties
- topology-preservation terms

These should be described as optional framework extensions rather than as part
of the benchmark-default training contract unless a specific experiment section
depends on them.

## 3. Confidence, Abstention, And Optional KNN Parameters

Repository-facing prediction controls currently include:

- `confidence_high`
- `confidence_low`
- `margin_threshold`
- `knn_correction`
- `knn_k`
- `knn_conf_low`
- `knn_vote_mode`
- `knn_reference_mode`
- `knn_index_mode`

Main-text recommendation:

- describe confidence-based routing and abstention as core behaviors
- describe KNN-assisted refinement only as an optional extension if it is
  needed for completeness
- keep exact parameter names and thresholds in supplement or repository-linked
  methods notes

## 4. Hierarchy Enforcement Detail

The current implementation supports post-prediction hierarchy enforcement using
explicit parent-child rules. When enabled, inconsistent child predictions can
be reset to `Unknown`.

Main-text recommendation:

- mention hierarchy-aware consistency enforcement as an active multi-level
  mechanism in the current benchmark path
- keep exact rule schema and implementation details out of the main Methods

## 5. Artifact Layout Detail

Preferred repository-facing artifact layout:

- `model.pth`
- `model_metadata.pkl`
- `model_reference.pkl`
- `model_manifest.json`

Main-text recommendation:

- describe AtlasMTL as exporting a portable trained-model bundle
- keep explicit filenames in supplement, software documentation, or response
  notes

## 6. Training-Default Provenance

The current software default is backed by completed optimization rounds:

- `2026-03-07_atlasmtl_param_lock_benchmark`
- `2026-03-09_atlasmtl_low_cost_optimization`

Important distinction:

- software default: `AdamW + weight_decay=5e-5`
- manuscript-grade formal comparison rows: retained third-wave AtlasMTL
  baseline rows

This distinction should remain explicit whenever implementation defaults are
discussed in supplement or reviewer-facing material.

## 7. Formal Refresh Disclosure

The fifth-round AtlasMTL-only formal refresh is best treated as supplementary
methodological context.

Recommended wording:

- the fifth-round formal refresh did not justify replacing the retained
  third-wave manuscript-grade AtlasMTL baseline rows
- the refreshed configuration remains the software default because it remains
  acceptable as a lightweight training default

## 8. Suggested Main-Text vs Supplement Split

Keep in main Methods:

- method overview and positioning
- shared encoder plus multi-head formulation
- weighted multi-task objective
- data contract and preprocessing logic
- current software default
- confidence / abstention behavior at a conceptual level
- hierarchy-aware consistency at a conceptual level
- benchmark task framing and primary metrics

Move to supplement or implementation details:

- exact hidden-layer widths and dropout values
- exact artifact filenames
- explicit threshold parameter names and defaults for prediction routing
- optional KNN parameter names and legacy routing controls
- exact hierarchy rule schema
- extended objective terms and optional training branches

## 9. Sixth-Round Multi-Level Interpretation Boundary

The completed sixth-round multi-level benchmark should be treated as a
capability round rather than a manuscript-table replacement round.

Current supplement-safe reading:

- the strongest positive result is strict hierarchy consistency under
  `enforce_hierarchy=True`
- the current multi-level `v1` setting does not uniformly outperform the
  retained single-level formal AtlasMTL baseline on finest-level headline
  metrics
- fine-level task weighting appears high-impact, but the current probes do not
  yet justify a single universal benchmark-default weight schedule

Recommended wording boundary:

- main Methods may describe AtlasMTL as supporting weighted multi-task learning
- supplement may note that benchmark-default multi-level task weighting remains
  under refinement after the sixth-round capability study
