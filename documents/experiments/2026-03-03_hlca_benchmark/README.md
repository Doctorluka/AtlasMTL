# HLCA Benchmark Pilot (2026-03-03)

This dossier is the pilot benchmark track for:

- reference dataset: `HLCA_Core`
- normalized inventory: `data_registry/reference_data_inventory_2026-03-03.md`

Scope for the first review round:

- formal `reference_heldout` benchmark on `ann_level_5`
- optional follow-up hierarchical run on `ann_level_1..ann_level_5`
- optional external query validation against `hlca_query_GSE302339`

Key contract note:

- `ann_finest_level` is not part of the cleaned runtime contract
- formal counts source is `layers["counts"]`
- `adata.X` is log-normalized and should not be treated as raw counts

Directory layout:

- `plan/`
- `manifests/`
- `scripts/`
- `notes/`
- `results_summary/`

Output root convention:

- `~/tmp/atlasmtl_benchmarks/2026-03-03/reference_heldout/HLCA_Core/`
- `~/tmp/atlasmtl_benchmarks/2026-03-03/external_query_validation/HLCA_Core/`
