# PH-Map first-wave preparation experiment record

Date: `2026-03-03`

## Design intent

Materialize a first executable `reference_heldout` asset pair for PH-Map under
the trial first-wave protocol:

- `5k` reference build subset
- `1k` heldout predict subset
- `sample`-aware external split
- canonical versionless Ensembl IDs
- reference-derived `3000` HVG panel

## Key parameters

- dataset: `PHMap_Lung_Full_v43_light`
- source path:
  `/home/data/fhz/project/phmap_package/data/real_test/phmap/phmap_clean.h5ad`
- manifest:
  `documents/experiments/2026-03-03_phmap_benchmark/manifests/reference_heldout/PHMap_Lung_Full_v43_light__anno_lv4__group_split_v1.yaml`
- split key: `sample`
- domain key: `sample`
- target label: `anno_lv4`
- build size: `5000`
- predict size: `1000`
- seed: `2026`
- candidate count: `128`
- counts layer: `counts`
- var names type: `symbol`
- Ensembl source column: `ENSEMBL`
- symbol source column: `Symbol`
- feature space: `hvg`
- HVG method: `seurat_v3`
- `n_top_genes`: `3000`

## Environment

- Python:
  `/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python`
- `NUMBA_CACHE_DIR`:
  `/home/data/fhz/project/phmap_package/atlasmtl/.tmp/numba_cache`

## Issues encountered

1. The first PH-Map preparation run used mapping-table mode instead of the
   explicit `ENSEMBL` column.
2. Root cause: explicit Ensembl-column detection treated placeholder strings
   like `"nan"` as invalid values rather than missing values.
3. Resolution:
   - updated `atlasmtl/preprocess/gene_ids.py`
   - explicit Ensembl source-column validation now ignores common missing-value
     placeholders (`"", "nan", "none", "na", "null"`)
   - added regression coverage in
     `tests/unit/test_preprocess_gene_ids.py`
   - reran the PH-Map preparation asset generation

## Non-blocking warnings

- numba reported that the TBB threading layer was disabled because the local TBB
  interface version is below the preferred threshold
- this did not block preprocessing or file generation

## Next-round recommendations

- keep this asset for first-wave full-pipeline smoke runs
- do not use the current `5k/1k` subset as a formal fine-label accuracy result
  without adding label-support constraints or stratified subset sampling
- if PH-Map remains a flagship benchmark, add a next-round split policy that
  enforces a minimum heldout per-label count
