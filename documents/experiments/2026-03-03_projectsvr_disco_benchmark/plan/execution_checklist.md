# DISCO pilot execution checklist

Date: `2026-03-03`

## Scenario identity

- Dataset ID: `DISCO_hPBMCs`
- Primary scenario: `reference_heldout`
- Primary target label: `cell_subtype`
- Split field: `sample`
- Locked domain key: `sample`
- Locked first-wave sizes: `5k` reference build / `1k` prediction

## Pre-run locking

- [ ] Confirm heldout split rule and seed
- [ ] Confirm exact `5k` build subset and `1k` prediction subset construction
- [ ] Confirm explicit preprocessing declaration for count-like `adata.X`
- [ ] Confirm counts-detection gate for `adata.X`
- [ ] Confirm whether coarse-level `cell_type` benchmark should ship with the first pass

## First pilot outputs

- [ ] single-level heldout manifest on `cell_subtype`
- [ ] preprocessing summary note
- [ ] output-root naming note
- [ ] results summary skeleton
- [ ] report and discussion note
- [ ] comparator fairness note

## Follow-up

- [ ] coarse-level heldout manifest on `cell_type`
- [ ] external query validation manifest for `pbmc_query`
- [ ] count-in-`adata.X` compatibility note
