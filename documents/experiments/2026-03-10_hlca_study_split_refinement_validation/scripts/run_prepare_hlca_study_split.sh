#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python}"
CONFIG_PATH="${REPO_ROOT}/documents/experiments/2026-03-10_hlca_study_split_refinement_validation/configs/hlca_study_split.yaml"

export NUMBA_CACHE_DIR="${REPO_ROOT}/.tmp/numba_cache"
export PYTHONPATH="${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/.tmp/numba_cache"

"${PYTHON_BIN}" \
  "${REPO_ROOT}/documents/experiments/2026-03-10_hlca_study_split_refinement_validation/scripts/prepare_hlca_study_split.py" \
  --config "${CONFIG_PATH}"
