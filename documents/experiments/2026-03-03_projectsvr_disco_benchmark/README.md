# ProjectSVR DISCO Benchmark Pilot (2026-03-03)

This dossier is the pilot benchmark track for:

- reference dataset: `DISCO_hPBMCs`
- normalized inventory: `data_registry/reference_data_inventory_2026-03-03.md`

Scope for the first review round:

- formal `reference_heldout` benchmark on `cell_subtype`
- optional coarse-level benchmark on `cell_type`
- optional external query validation against `pbmc_query`

Key contract note:

- this pilot is also the first explicit validation case for the
  “counts stored in `adata.X` without `layers["counts"]`” path

Directory layout:

- `plan/`
- `manifests/`
- `scripts/`
- `notes/`
- `results_summary/`

Output root convention:

- `~/tmp/atlasmtl_benchmarks/2026-03-03/reference_heldout/DISCO_hPBMCs/`
- `~/tmp/atlasmtl_benchmarks/2026-03-03/external_query_validation/DISCO_hPBMCs/`
