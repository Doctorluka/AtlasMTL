# PH-Map Benchmark Pilot (2026-03-03)

This dossier is the pilot benchmark track for:

- reference dataset: `PHMap_Lung_Full_v43_light`
- normalized inventory: `data_registry/reference_data_inventory_2026-03-03.md`

Scope for the first review round:

- formal `reference_heldout` benchmark on `anno_lv4`
- optional follow-up hierarchical run on `anno_lv1..anno_lv4`
- optional external query validation against `query_PH`

Directory layout:

- `plan/`
  - dataset-specific execution checklist and review notes
- `manifests/`
  - scenario manifests for this reference only
- `scripts/`
  - run helpers for this dossier only
- `notes/`
  - exclusions, dataset caveats, and split decisions
- `results_summary/`
  - compact summaries pointing to large runtime outputs under `~/tmp/`

Output root convention:

- `~/tmp/atlasmtl_benchmarks/2026-03-03/reference_heldout/PHMap_Lung_Full_v43_light/`
- `~/tmp/atlasmtl_benchmarks/2026-03-03/external_query_validation/PHMap_Lung_Full_v43_light/`
