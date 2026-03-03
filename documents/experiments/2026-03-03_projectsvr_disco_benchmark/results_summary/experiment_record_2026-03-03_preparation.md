# DISCO first-wave preparation experiment record

Date: `2026-03-03`

## Design intent

Materialize a first executable `reference_heldout` asset pair for
`DISCO_hPBMCs` under the trial first-wave protocol:

- `5k` reference build subset
- `1k` heldout predict subset
- `sample`-aware external split
- `adata.X` counts validation followed by counts-layer standardization
- canonical versionless Ensembl IDs
- reference-derived `3000` HVG panel

## Key parameters

- dataset: `DISCO_hPBMCs`
- source path:
  `/home/data/fhz/project/phmap_package/data/real_test/ProjectSVR/reference_atlas/DISCO_hPBMCs.h5ad`
- manifest:
  `documents/experiments/2026-03-03_projectsvr_disco_benchmark/manifests/reference_heldout/DISCO_hPBMCs__cell_subtype__group_split_v1.yaml`
- split key: `sample`
- domain key: `sample`
- target label: `cell_subtype`
- build size: `5000`
- predict size: `1000`
- seed: `2026`
- candidate count: `128`
- input matrix type: `counts`
- counts layer target: `counts`
- feature space: `hvg`
- HVG method: `seurat_v3`
- `n_top_genes`: `3000`

## Environment

- Python:
  `/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python`
- `NUMBA_CACHE_DIR`:
  `/home/data/fhz/project/phmap_package/atlasmtl/.tmp/numba_cache`

## Issues encountered

- no blocking preprocessing errors occurred during the DISCO first-wave
  preparation run

## Non-blocking warnings

- numba reported that the TBB threading layer was disabled because the local TBB
  interface version is below the preferred threshold
- this did not block preprocessing or file generation

## Next-round recommendations

- keep this asset for first-wave benchmark-runner smoke validation across all
  comparator wrappers
- add label-support aware subset materialization before promoting DISCO heldout
  runs into formal subtype-accuracy reporting
- review the symbol-to-Ensembl drop rate (`33538 -> 22500`) and decide whether a
  more atlas-specific mapping source is needed for later formal runs
