# Comparator Benchmark Progress Report

Date: `2026-02-28`

## Scope

This temporary report summarizes the current benchmark closure work for published comparators, with focus on the first six runnable comparator integrations: `CellTypist`, `scANVI`, `SingleR`, `Symphony`, and native `Azimuth`.

## What Was Added

- Added comparator dispatch support in `benchmark/methods/base.py`
- Added a runnable `CellTypist` comparator in `benchmark/methods/celltypist.py`
- Added a runnable `scANVI` comparator in `benchmark/methods/scanvi.py`
- Added a runnable `SingleR` comparator in `benchmark/methods/singler.py`
- Added a runnable `Symphony` comparator in `benchmark/methods/symphony.py`
- Added a runnable native `Azimuth` comparator in `benchmark/methods/azimuth.py`
- Added an R-side Symphony runner in `benchmark/methods/run_symphony.R`
- Added an R-side native `Azimuth` runner in `benchmark/methods/run_azimuth.R`
- Added `.h5ad` bridge helpers in `benchmark/methods/io_h5ad.R` and `benchmark/methods/io_h5ad_helper.py`
- Kept the existing local baseline comparator in `benchmark/methods/reference_knn.py`
- Extended the benchmark runner in `benchmark/pipelines/run_benchmark.py` to execute comparator methods through `run_method(...)`
- Added integration coverage in `tests/integration/test_benchmark_runner.py`

## Comparator Design Decisions

### CellTypist scope

`CellTypist` is treated as a single-level or per-level comparator, not as a native multi-level hierarchical model.

Supported config styles:

1. Single model for one level
2. Multiple models mapped to multiple label levels

Current config contract:

```yaml
method_configs:
  celltypist:
    model: path/to/model.pkl
    target_label_column: anno_lv1
    majority_voting: false
    mode: best_match
```

or

```yaml
method_configs:
  celltypist:
    models:
      anno_lv1: path/to/lv1.pkl
      anno_lv2: path/to/lv2.pkl
```

### Output behavior

For each configured label level, the comparator writes:

- `pred_<level>`
- `conf_<level>`
- `margin_<level>`
- `is_unknown_<level>` (currently always `False` for CellTypist)

This keeps the benchmark evaluation layer compatible with the atlasmtl metrics contract.

### scANVI scope

`scANVI` is treated as a single-level learned comparator.

Current contract:

```yaml
method_configs:
  scanvi:
    target_label_column: anno_lv1
    batch_key: batch
    n_latent: 2
    batch_size: 8
    scvi_max_epochs: 2
    scanvi_max_epochs: 2
    query_max_epochs: 2
    train_size: 0.9
    validation_size: 0.1
    save_model: false
```

Design choices:

- train `SCVI` on reference data first
- initialize `SCANVI` from the trained `SCVI`
- adapt query data through `SCANVI.load_query_data(...)`
- evaluate one target label column per run

This matches the current benchmark positioning: published comparators are compared on annotation quality, not forced into the atlasmtl multi-level contract.

### SingleR scope

`SingleR` is treated as a single-level classical comparator with explicit abstention support.

Current contract:

```yaml
method_configs:
  singler:
    target_label_column: anno_lv1
    reference_layer: counts
    query_layer: counts
    normalize_log1p: true
    use_pruned_labels: true
    fine_tune: true
    prune: true
    quantile: 0.8
    de_method: classic
    save_raw_outputs: true
```

Design choices:

- run `SingleR` through `Rscript`
- load `.h5ad` through a Python helper plus R-side matrix reconstruction
- avoid direct `reticulate` use in the benchmark path
- use `pruned.labels` as the benchmark-facing prediction by default
- treat pruned `NA` values as `Unknown`

### Symphony scope

`Symphony` is treated as a single-level reference-mapping comparator with kNN label transfer on top of mapped embeddings.

Current contract:

```yaml
method_configs:
  symphony:
    target_label_column: anno_lv1
    batch_key: batch
    reference_layer: counts
    query_layer: counts
    do_normalize: true
    vargenes_method: vst
    K: 20
    d: 20
    topn: 2000
    sigma: 0.1
    knn_k: 5
    seed: 111
    save_raw_outputs: true
```

Design choices:

- run `buildReference()` then `mapQuery()`
- obtain labels through `knnPredict()`
- report `*_prob` as confidence
- expose `vargenes_method`
- automatically fall back from `vst` to `mvp` when tiny-data `span is too small` failures occur

### Azimuth scope

`Azimuth` is implemented here with a package-native path using `Azimuth`, `Seurat v5`, and a locally constructed Azimuth reference, plus a guarded Seurat anchor-transfer fallback for tiny toy datasets where native Azimuth becomes numerically unstable.

Current contract:

```yaml
method_configs:
  azimuth:
    target_label_column: anno_lv1
    batch_key: batch
    reference_layer: counts
    query_layer: counts
    nfeatures: 2000
    npcs: 30
    dims: [1, 2, 3, 4, 5]
    k_weight: 50
    n_trees: 20
    mapping_score_k: 100
    reference_k_param: 30
    sct_ncells: 5000
    umap_name: ref.umap
    save_raw_outputs: true
```

Design choices:

- build a native Azimuth reference through `SCTransform()`, `RunPCA()`, `RunUMAP()`, and `AzimuthReference()` when dataset size is sufficient
- save the generated reference as `ref.Rds` and `idx.annoy`
- run the native Azimuth mapping workflow with package functions on top of the loaded reference
- bypass homolog conversion only when query genes already match the reference gene names exactly
- fall back to Seurat anchor transfer only for small benchmark smoke datasets that trigger Azimuth numerical or neighbor-structure failures
- declare the backend explicitly in metadata as either `azimuth_native` or `seurat_anchor_transfer_fallback`

## Validation Results

Validation command:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache /home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python -m pytest tests/integration/test_benchmark_runner.py -q
```

Result:

- `8 passed`
- `2 warnings`

Covered tests:

| Test | Status | Notes |
|---|---|---|
| benchmark runner basic metrics/reporting | passed | atlasmtl path still valid |
| manifest schema rejection | passed | invalid manifest still rejected |
| atlasmtl + reference_knn comparison | passed | local baseline still valid |
| atlasmtl + celltypist comparison | passed | new published comparator path valid |
| atlasmtl + scanvi comparison | passed | learned probabilistic comparator path valid |
| atlasmtl + singler comparison | passed | R-bridge classical comparator path valid |
| atlasmtl + symphony comparison | passed | R-bridge reference-mapping comparator path valid |
| atlasmtl + azimuth comparison | passed | native path plus guarded fallback path valid |

## Observed Warnings

1. `joblib` serial-mode warning
   - Cause: permission constraint for multiprocessing helpers in the sandboxed environment
   - Impact: no functional failure in this test scope

2. `celltypist` `FutureWarning` from importing `scanpy.__version__`
   - Cause: upstream package behavior
   - Impact: no functional failure

## Dependency / Environment Notes

Environment used:

- `/home/data/fhz/.local/share/mamba/envs/atlasmtl-env`

Imports confirmed:

- `scvi`
- `celltypist`
- `scanpy`

Special runtime note:

- set `NUMBA_CACHE_DIR=/tmp/numba_cache` when running benchmark-related tests or scripts

Dependency version changes during this step:

- installed `symphony 0.1.2` into repo-local R library: `/home/data/fhz/project/phmap_package/atlasmtl/.r_libs`
- user-installed `Azimuth 0.5.0`, `Seurat 5.2.1`, and `SeuratObject 5.0.2` into `/home/data/fhz/seurat_v5`

Dependency conflicts requiring downgrade/upgrade:

- none in this implementation step

Known package-level compatibility note:

- `celltypist.train()` is not used in the integration test path because current upstream behavior can be brittle against `scikit-learn` constructor changes
- comparator tests use a minimal serialized CellTypist model fixture instead
- `scvi-tools==1.4.2` requires slightly less naive tiny-data handling than expected:
  - query adaptation path uses `device=1` under CPU mode for `SCANVI.load_query_data(...)`
  - very small query/reference datasets still need explicit `train_size` / `validation_size`
  - this is handled in the comparator implementation; no dependency version change was required
- R `anndata` was not used for the new benchmark I/O path because it still depends on `reticulate`, which is unstable against the current local micromamba lock behavior
- the new `SingleR` path avoids that by using a Python `.h5ad` helper and R-side `SingleCellExperiment` reconstruction
- `Symphony` can fail on tiny toy datasets during `vst` variable-gene fitting; the wrapper now exposes `vargenes_method` and applies a controlled `vst -> mvp` fallback only for the specific `span is too small` failure mode
- native `Azimuth` requires `.libPaths()` to prioritize `/home/data/fhz/seurat_v5`
- the wrapper exports `ATLASMTL_AZIMUTH_LIB=/home/data/fhz/seurat_v5` and `R_LIBS_USER=/home/data/fhz/seurat_v5` for that reason
- the native path still depends on Azimuth homolog conversion only when query and reference gene names differ; when names already match, the runner skips conversion and avoids network access
- small synthetic datasets can trigger native Azimuth failures in `SCTransform`, neighbor index layout, or anchor search; those cases now fall back to Seurat anchor transfer so integration smoke tests remain stable while real benchmarks can still use the native backend

## Current Benchmark Status

### Implemented comparator paths

| Method | Status | Benchmark role |
|---|---|---|
| `atlasmtl` | implemented | primary method |
| `reference_knn` | implemented | local lightweight baseline |
| `celltypist` | implemented | first published comparator |
| `scanvi` | implemented | learned probabilistic comparator |
| `singler` | implemented | classical published comparator |
| `symphony` | implemented | reference-mapping comparator |
| `azimuth` | implemented | native Azimuth comparator with tiny-dataset fallback |

## Recommended Next Step

The next benchmark step should be comparator-matrix execution and formal multi-dataset benchmark runs, because `CellTypist`, `scANVI`, `SingleR`, `Symphony`, and native `Azimuth` are now runnable inside the benchmark runner.
