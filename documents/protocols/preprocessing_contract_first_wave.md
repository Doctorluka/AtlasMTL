# First-wave preprocessing contract

Date: `2026-03-03`

This document locks the preprocessing boundary for the first benchmark wave.
The goal is to make the pipeline reproducible and auditable before adding more
dataset-specific flexibility.

## Scope

First-wave execution scope:

- `PHMap_Lung_Full_v43_light`
  - `5k` reference build subset
  - `1k` prediction subset
  - standard path with `layers["counts"]`
- `DISCO_hPBMCs`
  - `5k` reference build subset
  - `1k` prediction subset
  - count-like `adata.X` path with no upstream `layers["counts"]`

Primary purpose:

- run the full workflow end-to-end
- standardize outputs and reporting
- validate preprocessing contracts
- require only sanity-level benchmark quality in this round

## Pipeline order

The first-wave preprocessing order is locked as:

1. load input AnnData
2. detect and validate raw-count contract
3. if needed, materialize `layers["counts"]`
4. canonicalize gene IDs
5. materialize split pools and subset sizes
6. write prepared assets under `~/tmp/`
7. run the benchmark only on prepared assets

The benchmark runner should not be the component that interprets ambiguous raw
inputs directly.

## Counts detection policy

Counts detection is the first required check for this round.

### Decision states

- `counts_confirmed`
  - safe to standardize into `layers["counts"]`
- `counts_suspected`
  - ambiguous; stop and require review
- `not_counts`
  - incompatible with this path; stop and require a valid counts layer

### Recommended first-wave checks

- matrix is non-negative
- nonzero values are overwhelmingly integer-like
- no meaningful mass of tiny positive fractional values
- minimum nonzero value is consistent with counts when present

### Recommended operational thresholds

- integer-likeness check:
  - at least `99.9%` of sampled nonzero values differ from the nearest integer
    by less than `1e-6`
- tiny-positive check:
  - tiny positive fraction among nonzero entries should remain near zero
- if `min_nonzero >= 1` and the integer-likeness test passes:
  - classify as `counts_confirmed`

### First-wave action rule

- if `counts_confirmed` and `layers["counts"]` already exists:
  - keep `layers["counts"]` as the formal counts contract
- if `counts_confirmed` and `layers["counts"]` is missing:
  - copy `adata.X` into `layers["counts"]`
- if `counts_suspected` or `not_counts` and no valid counts layer exists:
  - fail fast and record the reason

## Gene ID canonicalization policy

Gene canonicalization happens **after** counts detection and **before** any
split materialization.

The canonical internal namespace for formal experiments remains:

- versionless Ensembl IDs

The preprocessing module should expose options instead of hard-coding one
single conversion path. At minimum, it should allow the user to specify:

- `var_names_type`
  - `ensembl`, `ensembl_versioned`, `symbol`, `mixed`, `infer`
- `canonical_target`
  - default `ensembl`
- `ensembl_source_column`
  - such as `ENSEMBL` or `gene_ids`
- `symbol_source_column`
  - when symbols are not stored in `var_names`
- `mapping_table`
  - path or resource identifier
- `species`
  - `human`, `mouse`, `rat`, or explicit user override
- `duplicate_policy`
  - `sum`, `first`, `drop`, `error`
- `unmapped_policy`
  - `drop`, `keep_original`, `error`
- `version_strip`
  - whether to strip Ensembl suffixes such as `.7`
- `report_unmapped_top_n`
  - summary verbosity for logs/reports

## Canonicalization cases to support

### Case A: `var_names` already hold versionless Ensembl IDs

- action:
  - validate format
  - keep as canonical IDs

### Case B: `var_names` hold versioned Ensembl IDs

- action:
  - strip version suffixes
  - validate the stripped IDs
  - record that version stripping was applied

### Case C: `var_names` are symbols, but an Ensembl column exists

- action:
  - use the explicit Ensembl column as the canonical source
  - keep readable symbols in metadata
  - record the source column used

### Case D: only symbols are available

- action:
  - map symbols to Ensembl IDs via an explicit mapping resource
  - record:
    - mapping resource used
    - unmapped gene count
    - duplicate canonical ID count
    - duplicate handling policy
  - stop if `unmapped_policy=error` or `duplicate_policy=error`

## First-wave dataset-specific notes

### PH-Map

- counts path:
  - formal counts already live in `layers["counts"]`
- gene ID path:
  - treat as symbol-based input
  - expected canonical source is likely an existing metadata column such as
    `ENSEMBL`
- locked first-wave subset sizes:
  - `5k` reference build
  - `1k` prediction
- split key:
  - `sample`
- domain key:
  - `sample`

### DISCO_hPBMCs

- counts path:
  - detect whether `adata.X` is count-like
  - if confirmed, copy into `layers["counts"]`
- gene ID path:
  - currently symbol-based and may require mapping-table conversion
- locked first-wave subset sizes:
  - `5k` reference build
  - `1k` prediction
- split key:
  - `sample`
- domain key:
  - `sample`

## Output and reporting requirements

Prepared assets should be written under `~/tmp/` and should include:

- prepared reference subset
- prepared prediction subset
- preprocessing summary record
- counts-detection result
- gene-ID canonicalization summary

Repo-side dossier records under `documents/experiments/` should include:

- protocol note
- summary of what was executed
- summary of what failed or required manual override
- discussion points for the next round
- experiment record with key parameters, errors, and fixes
