#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python}"
MANIFEST="${REPO_ROOT}/documents/experiments/2026-03-05_formal_third_wave_hlca_benchmark/manifests/reference_heldout/HLCA_Core__ann_level_5__formal_runtime_cpu_train10k_test5k_v2.yaml"
OUT_DIR="${OUT_DIR:-/tmp/atlasmtl_benchmarks/2026-03-05/reference_heldout/HLCA_Core/benchmark/formal_train10k_test5k/cpu_group_v2_locked}"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export NUMBA_CACHE_DIR="${REPO_ROOT}/.tmp/numba_cache"
export PYTHONPATH="${REPO_ROOT}"
export ATLASMTL_FAIRNESS_POLICY="cpu_only_strict"

mkdir -p "${REPO_ROOT}/.tmp/numba_cache"
mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" "${REPO_ROOT}/documents/experiments/common/run_reference_heldout_scaleout_benchmark.py" \
  --dataset-manifest "${MANIFEST}" \
  --output-dir "${OUT_DIR}" \
  --device cpu \
  --methods atlasmtl reference_knn celltypist singler symphony seurat_anchor_transfer
