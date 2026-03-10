#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python}"
PREP_SCRIPT="${REPO_ROOT}/documents/experiments/2026-03-06_formal_third_wave_scaling/scripts/prepare_formal_third_wave_scaling_inputs.py"
DATASET_CONFIG="${REPO_ROOT}/documents/experiments/2026-03-09_multilevel_annotation_benchmark/configs/phmap_study_split.yaml"
OUT_ROOT="${OUT_ROOT:-/tmp/atlasmtl_benchmarks/2026-03-09/multilevel_annotation_study_split}"

export NUMBA_CACHE_DIR="${REPO_ROOT}/.tmp/numba_cache"
export PYTHONPATH="${REPO_ROOT}"

mkdir -p "${REPO_ROOT}/.tmp/numba_cache"

"${PYTHON_BIN}" "${PREP_SCRIPT}" \
  --dataset-config "${DATASET_CONFIG}" \
  --dataset-name PHMap_Lung_Full_v43_light \
  --output-root "${OUT_ROOT}"
