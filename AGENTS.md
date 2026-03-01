# Repository Guidelines

## Project Structure & Module Organization
This repository root is `/home/data/fhz/project/phmap_package/atlasmtl`. The Python package lives in `atlasmtl/`. Keep public entrypoints thin in `atlasmtl/core/api.py`; training, inference, runtime helpers, data transforms, and typed result/model containers now live in `atlasmtl/core/train.py`, `predict.py`, `runtime.py`, `data.py`, and `types.py`. Put confidence and KNN logic in `atlasmtl/mapping/`, AnnData writeback/export helpers in `atlasmtl/io/`, artifact serialization/manifest handling in `atlasmtl/models/`, plotting in `atlasmtl/pl/`, and reusable monitoring/progress helpers in `atlasmtl/utils/`. Benchmark protocol and runner skeleton belong in `benchmark/`, design docs in `documents/design/`, planning notes in `plan/`, and tests under `tests/unit`, `tests/integration`, and `tests/regression`. Do not modify `vendor/phmap_snapshot/`.

## Build, Test, and Development Commands
- `pip install -e .` installs the package in editable mode.
- `pip install -e .[dev]` installs development dependencies (`pytest`, `black`, `flake8`).
- `python -m compileall atlasmtl` runs a fast syntax check.
- `/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python -m pytest tests/unit/test_core_api.py -q` runs the focused unit API checks.
- `NUMBA_CACHE_DIR=/tmp/numba_cache /home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python -m pytest tests/integration/test_predict_integration.py -q` runs the end-to-end train/predict integration checks.
- `/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python -m pytest tests/integration/test_cli_smoke.py -q` checks the train/predict CLI wrappers and manifest loading.
- `flake8 atlasmtl tests` runs lint checks.
- `black atlasmtl tests scripts` formats code.

Use Python 3.8+ as defined in `pyproject.toml`.

Environment note:
- Target dev env: `/home/data/fhz/.local/share/mamba/envs/atlasmtl-env` (Python 3.11.14, phmap 0.1.1, scanpy 1.11.5 as of 2026-02-28).
- If `scanpy` import fails with a numba cache locator/write error, set `NUMBA_CACHE_DIR` to a writable path (e.g. `/tmp/numba_cache`) before running Python.
- Comparator benchmark R libraries:
  - native `Azimuth` / `Seurat v5`: `/home/data/fhz/seurat_v5`
  - repo-local comparator R packages such as `symphony`: `/home/data/fhz/project/phmap_package/atlasmtl/.r_libs`
- When changing comparator runners under `benchmark/methods/`, keep these library paths explicit in code or documentation instead of assuming system-default R libraries.

Model artifact note:
- Default recommendation is `reference_storage="external"` when building models.
- Preferred artifact layout is `model.pth` + `model_metadata.pkl` + `model_reference.pkl` + `model_manifest.json`.
- Use embedded reference storage only when a single self-contained file pair is more important than artifact size.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes, and explicit type hints for public APIs. Preserve the AnnData contract (`AnnData in, AnnData out`) and the established writeback names such as `obsm["X_pred_latent"]`, `obsm["X_pred_umap"]`, `obs["pred_<level>"]`, and `uns["atlasmtl"]`. Keep `PredictionResult` and `TrainedModel` methods stable; resource usage helpers (`get_resource_usage()` / `show_resource_usage()`) should remain lightweight and terminal-friendly.

## Research Positioning Constraints
Treat `atlasmtl` primarily as a reliable `sc -> sc reference mapping` and multi-level label transfer project, not as a general-purpose integrated embedding benchmark project. When making benchmark, architecture, or documentation changes, prioritize label accuracy, abstention quality, calibration, KNN rescue behavior, hierarchy consistency, robustness, and traceability over generic batch-correction or biological-conservation embedding metrics. Use scIB-style latent integration metrics only as secondary analysis unless the repository explicitly changes this positioning in `documents/design/research_positioning.md`. Do not expand the main benchmark to deconvolution, spatial localization, or gene-imputation tasks unless the user explicitly requests a scope expansion and the corresponding task-specific protocol is added.

## Gene ID And Feature Policy
- Treat versionless Ensembl IDs as the canonical internal gene namespace for training, prediction, and benchmark manifests.
- Preserve human-readable symbols in `adata.var["gene_symbol"]` or an equivalent metadata column, but do not rely on symbols as the sole alignment key in formal experiments.
- If an input dataset arrives with symbols only, require an explicit preprocessing step that records:
  - `var_names_type` such as `symbol` or `ensembl`
  - `species` such as `human`, `mouse`, or `rat`
  - the mapping resource or conversion table used to derive canonical Ensembl IDs
- Strip Ensembl version suffixes before alignment, and make duplicate/failed mappings explicit in metadata or protocol records rather than silently ignoring them.
- For formal runs, prefer a reference-derived HVG panel after canonicalization. Use whole-matrix training only as an explicit ablation or when resources and experiment design justify it.
- Bundled human/mouse/rat mapping baseline lives at `atlasmtl/resources/gene_id/biomart_human_mouse_rat.tsv.gz`; keep column names explicit and document any updates to that file.

## Benchmark Comparator Constraints
- Current runnable benchmark methods are `atlasmtl`, `reference_knn`, `celltypist`, `scanvi`, `singler`, `symphony`, and `azimuth`.
- Treat most external comparators as single-level baselines unless they natively support a richer contract; do not overstate them as full multi-level hierarchical methods.
- For formal benchmark interpretation, compare all methods on a shared target label level first, then report atlasmtl-specific hierarchy/KNN/open-set behavior as secondary method-specific analysis.
- `azimuth` should prefer the native backend when dataset size and numerics permit. Any Seurat anchor-transfer fallback must be clearly labeled as fallback in metadata and documentation, and should not be presented as the primary formal benchmark result.

## Testing Guidelines
Write unit tests for model utilities, serialization, and confidence/KNN logic in `tests/unit/`, integration tests for end-to-end AnnData IO and CLI behavior in `tests/integration/`, and metric/regression checks in `tests/regression/`. Include at least one smoke test for `build_model()` and `predict()` whenever API signatures, artifact layout, or `uns["atlasmtl"]` metadata change.

## Commit & Pull Request Guidelines
Use Conventional Commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`. The active branch is `main` tracking `origin/main`; run commits and pushes from this repository root. Keep PRs scoped, include a short problem statement, list validation commands run, and describe any changes to output fields in `obs/obsm/uns`, runtime summaries, or model artifact filenames. Avoid rewriting shared history unless explicitly required; if history rewrite is unavoidable, use `--force-with-lease` and document the reason in the PR.
