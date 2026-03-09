#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python}"
RUNNER="${REPO_ROOT}/documents/experiments/common/run_reference_heldout_scaleout_benchmark.py"
MANIFEST_DIR="${REPO_ROOT}/documents/experiments/2026-03-06_formal_third_wave_scaling/manifests/reference_heldout"
OUT_ROOT="${OUT_ROOT:-/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/HLCA_Core/benchmark/sanity_check/cpu}"

BUILD_MANIFEST="${MANIFEST_DIR}/HLCA_Core__ann_level_5__formal_build_scaling_cpu_build100000_eval10k_v1.yaml"
PREDICT_MANIFEST="${MANIFEST_DIR}/HLCA_Core__ann_level_5__formal_predict_scaling_cpu_build100000_predict10000_v1.yaml"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export NUMBA_CACHE_DIR="${REPO_ROOT}/.tmp/numba_cache"
export PYTHONPATH="${REPO_ROOT}"
export ATLASMTL_FAIRNESS_POLICY="cpu_only_strict"

mkdir -p "${REPO_ROOT}/.tmp/numba_cache"
mkdir -p "${OUT_ROOT}"

echo "=== HLCA CPU sanity: build scaling (100k -> eval10k) ==="
"${PYTHON_BIN}" "${RUNNER}" \
  --dataset-manifest "${BUILD_MANIFEST}" \
  --output-dir "${OUT_ROOT}/build100k_eval10k" \
  --device cpu \
  --methods atlasmtl reference_knn celltypist singler symphony seurat_anchor_transfer

echo "=== HLCA CPU sanity: predict scaling (100k -> predict10k) ==="
"${PYTHON_BIN}" "${RUNNER}" \
  --dataset-manifest "${PREDICT_MANIFEST}" \
  --output-dir "${OUT_ROOT}/predict100k_10k" \
  --device cpu \
  --methods atlasmtl reference_knn celltypist singler symphony seurat_anchor_transfer
