# AnnData cleaning checklist (atlasmtl contract)

This checklist defines the **minimum** requirements for a dataset to be used
in formal atlasmtl training, prediction, and benchmark runs.

Primary protocol reference:

- `documents/protocols/experiment_protocol.md`

Design reference:

- `documents/design/preprocessing.md`

## A) Required matrix contract

- [ ] Raw counts are available as `adata.layers["counts"]` (or can be
      deterministically constructed and persisted there).
- [ ] `adata.X` semantics are **explicitly declared** for the run:
  - `input_matrix_type` in the dataset manifest (`counts` vs `lognorm` vs `infer`)
- [ ] If `adata.raw` exists, document whether it is:
  - raw counts (preferred), or
  - log-normalized / scaled data (not acceptable as counts)
- [ ] For comparator fairness, record which matrix each method consumes
      (via the benchmark `input_contract` table).

## B) Gene namespace and alignment

- [ ] Declare `var_names_type` (`symbol` vs `ensembl`) and `species`.
- [ ] Canonical internal namespace for formal runs: versionless Ensembl IDs.
- [ ] Strip Ensembl version suffixes (e.g. `.7`) before alignment.
- [ ] Preserve symbols in `adata.var["gene_symbol"]` when available.
- [ ] Record mapping resource:
  - path / version hash, and
  - unmapped + duplicate handling policy.
- [ ] After canonicalization, ensure reference/query share the same feature
      panel and order (atlasmtl requires exact feature alignment).

## C) Labels and hierarchy

- [ ] Define `label_columns` (single-level or multi-level) and ensure they
      exist in both reference and query `obs`.
- [ ] If multi-level labels exist, define `hierarchy_rules` and confirm that
      parent–child mappings are internally consistent.
- [ ] Decide Unknown handling policy:
  - closed-set only, or
  - abstention (`Unknown`) + threshold selection rule.

## D) Domain/shift metadata (for paper-grade evaluation)

- [ ] Provide a `domain_key` for grouped reporting when applicable (e.g.
      `study`, `dataset`, `sample`, `platform`).
- [ ] For shift stories, document the shift axis (platform/tissue/disease/batch
      /label-set shift) and how the split is constructed.

## E) Geometry assets (optional but strongly recommended)

- [ ] If evaluating coordinate heads, declare:
  - reference `coord_targets` (`obsm` keys), and
  - query `query_coord_targets` for scoring.
- [ ] If evaluating KNN in embedding space, declare the embedding to use and
      ensure it exists on reference/query (and document whether it is predicted
      or native to the dataset).
- [ ] If graph metrics are used, confirm `obsp["connectivities"]` and
      `obsp["distances"]` exist and have expected semantics.

## F) Minimal validation outputs (must be persisted per dataset)

- [ ] A short “data audit” markdown:
  - shapes, layers, `var_names_type`, missingness, label coverage, domain counts
- [ ] A preprocessing report (if preprocessing is applied):
  - unmapped genes, duplicate collapse counts, final feature panel size

