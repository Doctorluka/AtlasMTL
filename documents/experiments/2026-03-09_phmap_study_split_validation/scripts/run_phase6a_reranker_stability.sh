#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python}"

export PYTHONPATH="${REPO_ROOT}"
export NUMBA_CACHE_DIR="${REPO_ROOT}/.tmp/numba_cache"

mkdir -p "${REPO_ROOT}/.tmp/numba_cache"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_phase6a_reranker_stability.py"
