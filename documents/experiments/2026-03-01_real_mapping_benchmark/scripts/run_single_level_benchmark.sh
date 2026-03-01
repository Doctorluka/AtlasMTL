#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python"
NUMBA_CACHE_DIR="${REPO_ROOT}/.tmp/numba_cache"
RUNTIME_ROOT="/home/data/fhz/tmp/atlasmtl_real_mapping_benchmark_20260301"

mkdir -p "${NUMBA_CACHE_DIR}" "${RUNTIME_ROOT}/single_level_benchmark" "${RUNTIME_ROOT}/logs"
export NUMBA_CACHE_DIR
export PYTHONPATH="${REPO_ROOT}"
export ATLASMTL_PYTHON="${PYTHON_BIN}"
export ATLASMTL_AZIMUTH_LIB="/home/data/fhz/seurat_v5"
export R_LIBS_USER="/home/data/fhz/project/phmap_package/atlasmtl/.r_libs"

"${PYTHON_BIN}" "${REPO_ROOT}/benchmark/pipelines/run_benchmark.py" \
  --dataset-manifest "${REPO_ROOT}/documents/experiments/2026-03-01_real_mapping_benchmark/manifests/single_level_benchmark.yaml" \
  --output-dir "${RUNTIME_ROOT}/single_level_benchmark" \
  --methods atlasmtl reference_knn celltypist scanvi singler symphony azimuth \
  --device cpu
