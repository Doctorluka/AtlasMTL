# PH-Map pilot execution checklist

Date: `2026-03-03`

## Scenario identity

- Dataset ID: `PHMap_Lung_Full_v43_light`
- Primary scenario: `reference_heldout`
- Primary target label: `anno_lv4`
- Split field: `sample`
- Locked domain key: `sample`
- Locked first-wave sizes: `5k` reference build / `1k` prediction

## Pre-run locking

- [ ] Confirm heldout split rule and seed
- [ ] Confirm exact `5k` build subset and `1k` prediction subset construction
- [ ] Confirm `var_names_type` and Ensembl canonicalization policy
- [ ] Confirm counts detection record even though `layers["counts"]` exists

## First pilot outputs

- [ ] single-level heldout manifest on `anno_lv4`
- [ ] preprocessing summary note
- [ ] output-root naming note
- [ ] results summary skeleton
- [ ] report and discussion note
- [ ] comparator fairness note

## Follow-up

- [ ] hierarchical heldout manifest on `anno_lv1..anno_lv4`
- [ ] external query validation manifest for `query_PH`
- [ ] shift scenario design note
