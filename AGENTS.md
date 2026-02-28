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

Model artifact note:
- Default recommendation is `reference_storage="external"` when building models.
- Preferred artifact layout is `model.pth` + `model_metadata.pkl` + `model_reference.pkl` + `model_manifest.json`.
- Use embedded reference storage only when a single self-contained file pair is more important than artifact size.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes, and explicit type hints for public APIs. Preserve the AnnData contract (`AnnData in, AnnData out`) and the established writeback names such as `obsm["X_pred_latent"]`, `obsm["X_pred_umap"]`, `obs["pred_<level>"]`, and `uns["atlasmtl"]`. Keep `PredictionResult` and `TrainedModel` methods stable; resource usage helpers (`get_resource_usage()` / `show_resource_usage()`) should remain lightweight and terminal-friendly.

## Testing Guidelines
Write unit tests for model utilities, serialization, and confidence/KNN logic in `tests/unit/`, integration tests for end-to-end AnnData IO and CLI behavior in `tests/integration/`, and metric/regression checks in `tests/regression/`. Include at least one smoke test for `build_model()` and `predict()` whenever API signatures, artifact layout, or `uns["atlasmtl"]` metadata change.

## Commit & Pull Request Guidelines
Use Conventional Commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`. The active branch is `main` tracking `origin/main`; run commits and pushes from this repository root. Keep PRs scoped, include a short problem statement, list validation commands run, and describe any changes to output fields in `obs/obsm/uns`, runtime summaries, or model artifact filenames. Avoid rewriting shared history unless explicitly required; if history rewrite is unavoidable, use `--force-with-lease` and document the reason in the PR.
