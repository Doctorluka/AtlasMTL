# Dataset Registry (local, not committed)

This file records **local** large datasets available on this machine for
future atlasmtl cleaning + benchmarking work. These datasets are **not**
checked into git; only paths and minimal metadata are recorded here.

For the normalized, benchmark-facing working records of cleaned reference and
query datasets, use:

- `data_registry/reference_data_inventory_2026-03-03.md`
- `data_registry/query_data_inventory_2026-03-03.md`
- `data_registry/benchmark_scenario_registry_2026-03-03.md`

Status legend:

- `READY`: can be used in atlasmtl runs with minimal/no conversion
- `NEEDS_CLEANING`: expected to work after conversion to atlasmtl AnnData contract
- `UNKNOWN`: not yet inspected

## 1) PH-Map full lung atlas (v43 light)

- **ID**: `PHMap_Lung_Full_v43_light`
- **Path**: `/home/data/fhz/project/reference_map_PH/data/lung/43_lung_atlas_refine_tidy_light.h5ad`
- **Size**: ~2.30 GB
- **Shape**: `234,390 cells × 21,977 genes`
- **Gene namespace (observed)**: `symbol`-like (`var_names[:5]` includes `A1BG`)
- **Counts contract**:
  - `layers["counts"]`: present
  - `raw`: absent
- **Labels (observed)**: `obs["anno_lv1..anno_lv4"]` present
- **Embeddings / graphs (observed)**:
  - `obsm`: `X_pca`, `X_scVI`, `X_scANVI`, `X_umap`, `X_umap_scVI`, `X_umap_scANVI`
  - `obsp`: `connectivities`, `distances`
- **Suggested role**: primary “paper-grade” dataset for coordinate/KNN geometry + hierarchy consistency work.
- **Status**: `READY` (after defining the symbol→Ensembl canonicalization policy for formal runs)

**Cleaning / protocol TODOs (before formal benchmark)**

- Lock gene ID policy: record `var_names_type="symbol"` and the exact mapping resource used for Ensembl canonicalization.
- Confirm `layers["counts"]` semantics (raw UMI counts vs normalized).
- Confirm domain key(s): candidate `obs["sample"]`, `obs["dataset"]`, `obs["study"]`.

## 2) HLCA core (Human Lung Cell Atlas)

- **ID**: `HLCA_Core`
- **Path**: `/home/data/fhz/project/phmap_package/data/HLCA/hlca_core.h5ad`
- **Size**: ~5.47 GB
- **Shape**: `584,944 cells × 27,402 genes`
- **Gene namespace (observed)**: `ensembl` versionless (`ENSG...` in `var_names`)
- **Counts contract**:
  - `layers["counts"]`: missing (as loaded)
  - `raw`: present (`raw.shape == (584,944, 27,402)`)
- **Labels (observed)**:
  - multi-level: `obs["ann_level_1".."ann_level_4"]`
  - finest: `obs["ann_finest_level"]`
  - ontology-rich metadata (CZI cellxgene schema fields)
- **Embeddings / graphs (observed)**:
  - `obsm`: `X_scanvi_emb`, `X_umap`
  - `obsp`: `connectivities`, `distances`
- **Suggested role**: large-scale stress test for CPU-first scaling + multi-level label transfer under distribution shift.
- **Status**: `NEEDS_CLEANING`

**Cleaning / protocol TODOs (blocking)**

- Decide and document where raw counts live (likely `raw.X`): convert into `layers["counts"]` deterministically.
- Ensure `adata.X` interpretation is explicit (`input_matrix_type`) after conversion.
- Define label columns to use for atlasmtl (`ann_level_*` vs remapped `anno_lv*`) and record mapping.

## 3) ProjectSVR datasets (Seurat objects)

- **ID**: `ProjectSVR_Package`
- **Root path**: `/home/data/fhz/project/phmap_package/data/ProjectSVR/`
- **Size**: ~2.8 GB
- **Contents (observed)**:
  - `reference_atlas/`: `*.seurat.slim.qs` (10 files)
  - `query_data/`: `*.seurat.slim.qs` + `*.seurat.rds` (5 files)
  - `quick_start/`: small `*.seurat.slim.rds` tutorial pair
- **Suggested role**: comparator-aligned datasets for “ProjectSVR vs atlasmtl” gap analysis and narrative.
- **Status**: `NEEDS_CLEANING` (conversion required)

**Conversion / protocol TODOs (blocking)**

- Convert Seurat `.qs/.rds` to `.h5ad` with an explicit counts-layer contract:
  - ensure raw counts end up in `layers["counts"]`
  - record whether `adata.X` is counts or normalized (`input_matrix_type`)
- Standardize gene IDs (symbols vs Ensembl) and record `species`.
- For each pair: lock `label_columns` (single-level vs multi-level availability) and a `domain_key` if applicable.
