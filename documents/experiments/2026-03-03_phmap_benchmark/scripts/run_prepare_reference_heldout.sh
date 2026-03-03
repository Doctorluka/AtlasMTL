#!/usr/bin/env zsh
set -euo pipefail

REPO_ROOT="/home/data/fhz/project/phmap_package/atlasmtl"
PYTHON_BIN="/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python"
MANIFEST="${REPO_ROOT}/documents/experiments/2026-03-03_phmap_benchmark/manifests/reference_heldout/PHMap_Lung_Full_v43_light__anno_lv4__group_split_v1.yaml"
SOURCE_H5AD="/home/data/fhz/project/phmap_package/data/real_test/phmap/phmap_clean.h5ad"

mkdir -p "${REPO_ROOT}/.tmp/numba_cache"
export NUMBA_CACHE_DIR="${REPO_ROOT}/.tmp/numba_cache"
export PYTHONPATH="${REPO_ROOT}"

"${PYTHON_BIN}" "${REPO_ROOT}/documents/experiments/common/prepare_reference_heldout_first_wave.py" \
  --dataset-manifest "${MANIFEST}" \
  --source-h5ad "${SOURCE_H5AD}" \
  --split-key sample \
  --domain-key sample \
  --target-label anno_lv4 \
  --build-size 5000 \
  --predict-size 1000 \
  --seed 2026 \
  --n-candidates 128
