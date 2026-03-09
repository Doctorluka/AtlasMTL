#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python}"
RUNNER="${REPO_ROOT}/documents/experiments/common/run_reference_heldout_scaleout_benchmark.py"
MANIFEST_DIR="${REPO_ROOT}/documents/experiments/2026-03-06_formal_third_wave_scaling/manifests/reference_heldout"
OUT_ROOT="${OUT_ROOT:-/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/HLCA_Core/benchmark/formal_main/gpu}"

BUILD_SIZES=(10000 20000 30000 50000 100000 150000 200000 300000)
PREDICT_SIZES=(1000 3000 5000 8000 10000 15000 20000 50000)

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export NUMBA_CACHE_DIR="${REPO_ROOT}/.tmp/numba_cache"
export PYTHONPATH="${REPO_ROOT}"
export ATLASMTL_FAIRNESS_POLICY="mixed_backend_labeled"

mkdir -p "${REPO_ROOT}/.tmp/numba_cache"
mkdir -p "${OUT_ROOT}"

"${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("GPU preflight failed: torch.cuda.is_available() is false")
print("GPU preflight ok:", torch.cuda.get_device_name(0))
PY

for build_size in "${BUILD_SIZES[@]}"; do
  manifest="${MANIFEST_DIR}/HLCA_Core__ann_level_5__formal_build_scaling_gpu_build${build_size}_eval10k_v1.yaml"
  out_dir="${OUT_ROOT}/build_${build_size}_eval10k"
  echo "=== HLCA GPU build scaling: build=${build_size} -> eval10k ==="
  "${PYTHON_BIN}" "${RUNNER}" \
    --dataset-manifest "${manifest}" \
    --output-dir "${out_dir}" \
    --device cuda \
    --methods atlasmtl scanvi
done

for predict_size in "${PREDICT_SIZES[@]}"; do
  manifest="${MANIFEST_DIR}/HLCA_Core__ann_level_5__formal_predict_scaling_gpu_build100000_predict${predict_size}_v1.yaml"
  out_dir="${OUT_ROOT}/predict_100000_${predict_size}"
  echo "=== HLCA GPU predict scaling: build=100000 -> predict=${predict_size} ==="
  "${PYTHON_BIN}" "${RUNNER}" \
    --dataset-manifest "${manifest}" \
    --output-dir "${out_dir}" \
    --device cuda \
    --methods atlasmtl scanvi
done
