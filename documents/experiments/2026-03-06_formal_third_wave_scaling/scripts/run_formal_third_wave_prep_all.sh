#!/usr/bin/env zsh
set -euo pipefail

REPO_ROOT="/home/data/fhz/project/phmap_package/atlasmtl"
PYTHON_BIN="/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python"
CONFIG_PATH="${REPO_ROOT}/documents/experiments/2026-03-06_formal_third_wave_scaling/configs/datasets.yaml"
SCRIPT_PATH="${REPO_ROOT}/documents/experiments/2026-03-06_formal_third_wave_scaling/scripts/prepare_formal_third_wave_scaling_inputs.py"
OUTPUT_ROOT="/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave"

mkdir -p "${REPO_ROOT}/.tmp/numba_cache"
export NUMBA_CACHE_DIR="${REPO_ROOT}/.tmp/numba_cache"
export PYTHONPATH="${REPO_ROOT}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

datasets=(
  "HLCA_Core"
  "PHMap_Lung_Full_v43_light"
  "mTCA"
  "DISCO_hPBMCs"
  "Vento"
)

for dataset in "${datasets[@]}"; do
  echo "=== preparing ${dataset} ==="
  "${PYTHON_BIN}" "${SCRIPT_PATH}" \
    --dataset-config "${CONFIG_PATH}" \
    --dataset-name "${dataset}" \
    --output-root "${OUTPUT_ROOT}"
done
