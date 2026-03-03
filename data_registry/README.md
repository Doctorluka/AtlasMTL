# Data Registry

Store dataset manifests only (paths, version hashes, splits, label hierarchy mappings).
Do not commit raw large datasets.

TODO:
- Track local large datasets (paths + minimal metadata) in `data_registry/datasets.md`.
- Add dataset manifests after cleaning (train/val/test splits + OOD/novel-type scenarios).

Current recorded large datasets (local-only):

- PH-Map full lung atlas (`PHMap_Lung_Full_v43_light`)
- HLCA core (`HLCA_Core`)
- ProjectSVR package (Seurat objects)

See: `data_registry/datasets.md`.

Normalized reference-data inventory:

- `data_registry/reference_data_inventory_2026-03-03.md`

Normalized query-data inventory:

- `data_registry/query_data_inventory_2026-03-03.md`

Benchmark scenario registry:

- `data_registry/benchmark_scenario_registry_2026-03-03.md`

Cleaning acceptance checklist:

- `data_registry/cleaning_checklist.md`
