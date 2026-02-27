# Repository Guidelines

## Project Structure & Module Organization
`atlasmtl` is a Python package rooted at `atlasmtl/`. Core training and inference code lives in `atlasmtl/core/` (`api.py`, `model.py`). Keep mapping/correction logic in `atlasmtl/mapping/`, IO adapters in `atlasmtl/io/`, plotting in `atlasmtl/pl/`, and reusable helpers in `atlasmtl/utils/`. Benchmark code belongs under `benchmark/` (`datasets/`, `methods/`, `pipelines/`, `metrics/`, `reports/`). Design and paper assets are in `documents/`, runnable examples in `notebooks/`, and tests in `tests/` (`unit/`, `integration/`, `regression/`). `vendor/phmap_snapshot/` is a read-only reference snapshot and must not be edited for feature work.

## Build, Test, and Development Commands
- `pip install -e .` installs the package in editable mode.
- `pip install -e .[dev]` installs development dependencies (`pytest`, `black`, `flake8`).
- `python -m compileall atlasmtl` runs a fast syntax check.
- `pytest` runs tests; use `pytest tests/unit -q` for quick checks.
- `flake8 atlasmtl tests` runs lint checks.
- `black atlasmtl tests scripts` formats code.

Use Python 3.8+ as defined in `pyproject.toml`.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes, and explicit type hints for public APIs. Preserve the AnnData contract (`AnnData in, AnnData out`) and consistent field naming such as `obsm["X_pred_latent"]`, `obsm["X_pred_umap"]`, `obs["pred_<level>_final"]`, and `uns["atlasmtl"]`.

## Testing Guidelines
Write unit tests for model utilities and confidence/KNN logic in `tests/unit/`, integration tests for end-to-end AnnData IO in `tests/integration/`, and metric/regression checks in `tests/regression/`. Include at least one smoke test for `build_model()` and `predict()` whenever API signatures change.

## Commit & Pull Request Guidelines
Git history is currently empty, so adopt Conventional Commits from now on: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`. Keep PRs scoped, include a short problem statement, list validation commands run, and describe any changes to output fields in `obs/obsm/uns`.
