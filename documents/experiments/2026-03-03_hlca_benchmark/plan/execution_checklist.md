# HLCA pilot execution checklist

Date: `2026-03-03`

## Scenario identity

- Dataset ID: `HLCA_Core`
- Primary scenario: `reference_heldout`
- Primary target label: `ann_level_5`
- Split field: `donor_id`
- Current default domain key: `donor_id`

## Pre-run locking

- [ ] Confirm heldout split rule and seed
- [ ] Confirm admissible training sizes after split
- [ ] Confirm heldout evaluation sizes after split
- [ ] Confirm Ensembl canonicalization rule for mixed query-side IDs
- [ ] Confirm whether external `Gold` remains marker-only or receives special approval later

## First pilot outputs

- [ ] single-level heldout manifest on `ann_level_5`
- [ ] output-root naming note
- [ ] results summary skeleton
- [ ] comparator fairness note

## Follow-up

- [ ] hierarchical heldout manifest on `ann_level_1..ann_level_5`
- [ ] external query validation manifest for `hlca_query_GSE302339`
- [ ] donor/group shift scenario design note
