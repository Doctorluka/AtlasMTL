# Pilot benchmark review checklist

Date: `2026-03-03`

Use this checklist before executing any pilot benchmark manifest. The goal is
to catch contract drift early, before a run produces non-comparable outputs.

## 1) Dataset contract review

- [ ] Reference `.h5ad` path matches the normalized inventory
- [ ] Query or heldout-test `.h5ad` path matches the intended scenario
- [ ] Target label column exists and is populated
- [ ] Split/group field exists and is populated
- [ ] `domain_key` is explicitly chosen and justified for grouped reporting
- [ ] `var_names_type` is recorded correctly
- [ ] `species` is recorded correctly
- [ ] `input_matrix_type` matches observed matrix semantics
- [ ] `counts_layer` is present or will be materialized by preprocessing

## 2) Split fairness review

- [ ] Train / validation / heldout-test pools are group-aware
- [ ] No same-group leakage across pools
- [ ] First-wave subset sizes are explicitly locked (`5k` build / `1k` predict for current pilots)
- [ ] Heldout-test labels retain enough support for reporting
- [ ] Training-size and evaluation-size subsets come from fixed pools
- [ ] Skipped size points are documented explicitly

## 3) Comparator fairness review

- [ ] All methods use the same target label in this scenario
- [ ] All methods use the same heldout truth pool
- [ ] All first-wave comparators are included unless an exclusion is documented
- [ ] Batch/domain keys are recorded consistently
- [ ] Counts semantics are recorded consistently
- [ ] Any dataset-specific tuning is documented in the scenario note
- [ ] Any method-specific tuning beyond defaults is documented in the scenario note

## 4) Output traceability review

- [ ] Output root follows the benchmark naming convention
- [ ] Large runtime assets stay under `~/tmp/`
- [ ] Repo-side dossier will keep a report, summary, and discussion note
- [ ] `split_name` and `split_description` are specific enough to audit later
- [ ] `run_manifest.json` will be retained
- [ ] `metrics.json` and `summary.csv` are expected outputs
- [ ] `paper_tables/` export expectation is recorded

## 5) Scenario boundary review

- [ ] Scenario is clearly marked `reference_heldout` or `external_query_validation`
- [ ] No external query scenario is framed as formal accuracy benchmark by default
- [ ] Any use of author-provided query labels is described as visualization or review unless explicitly approved

## 6) Pilot sign-off record

- Dataset:
- Scenario ID:
- Manifest path:
- Reviewed by:
- Review date:
- Decision: `approved_for_execution` / `needs_revision`
- Notes:
