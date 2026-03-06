#!/usr/bin/env zsh
set -euo pipefail

REPO_ROOT="/home/data/fhz/project/phmap_package/atlasmtl"
PYTHON_BIN="/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python"

cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/.tmp/numba_cache"

export PYTHONPATH="${REPO_ROOT}"
export NUMBA_CACHE_DIR="${REPO_ROOT}/.tmp/numba_cache"

"${PYTHON_BIN}" documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/scripts/run_atlasmtl_param_sweep.py \
  --stage stage_b \
  --device cuda
