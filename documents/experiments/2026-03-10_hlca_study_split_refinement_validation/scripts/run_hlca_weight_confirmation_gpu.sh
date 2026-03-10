#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python}"
RUNNER="${REPO_ROOT}/benchmark/pipelines/run_benchmark.py"

MANIFEST_SCRIPT="${SCRIPT_DIR}/generate_hlca_weight_confirmation_manifests.py"
COLLECT_SCRIPT="${SCRIPT_DIR}/collect_hlca_weight_confirmation_results.py"
TRAIN_INDEX="${REPO_ROOT}/documents/experiments/2026-03-10_hlca_study_split_refinement_validation/manifests/weight_confirmation/train_manifest_index.json"
PREDICT_INDEX="${REPO_ROOT}/documents/experiments/2026-03-10_hlca_study_split_refinement_validation/manifests/weight_confirmation/predict_manifest_index.json"
TMP_ROOT="/tmp/atlasmtl_benchmarks/2026-03-10/hlca_study_split_weight_confirmation"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export NUMBA_CACHE_DIR="${REPO_ROOT}/.tmp/numba_cache"
export PYTHONPATH="${REPO_ROOT}"
export ATLASMTL_FAIRNESS_POLICY="mixed_backend_labeled"

mkdir -p "${REPO_ROOT}/.tmp/numba_cache"

has_run_outputs() {
  local out_dir="$1"
  [[ -f "${out_dir}/metrics.json" && -f "${out_dir}/summary.csv" ]]
}

gpu_preflight() {
  "${PYTHON_BIN}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("GPU preflight failed: torch.cuda.is_available() is false")
print("GPU preflight ok:", torch.cuda.get_device_name(0))
PY
}

gpu_preflight
"${PYTHON_BIN}" "${MANIFEST_SCRIPT}"

"${PYTHON_BIN}" - "${TRAIN_INDEX}" <<'PY' | while IFS=$'\t' read -r config_name manifest_path; do
import json
import sys
rows = json.load(open(sys.argv[1]))
for item in rows:
    print(f"{item['config_name']}\t{item['train_manifest_path']}")
PY
  [[ -n "${manifest_path}" ]] || continue
  run_dir="${TMP_ROOT}/train/${config_name}/runs/atlasmtl"
  mkdir -p "${run_dir}"
  if has_run_outputs "${run_dir}"; then
    echo "=== skip hlca weight train ${config_name} ==="
    continue
  fi
  echo "=== hlca weight train ${config_name} ==="
  "${PYTHON_BIN}" "${RUNNER}" \
    --dataset-manifest "${manifest_path}" \
    --output-dir "${run_dir}" \
    --device cuda \
    --methods atlasmtl \
    > "${run_dir}/stdout.log" \
    2> "${run_dir}/stderr.log"
done

"${PYTHON_BIN}" - "${PREDICT_INDEX}" <<'PY' | while IFS=$'\t' read -r config_name point manifest_path; do
import json
import sys
rows = json.load(open(sys.argv[1]))
for item in rows:
    print(f"{item['config_name']}\t{item['point']}\t{item['predict_manifest_path']}")
PY
  [[ -n "${manifest_path}" ]] || continue
  model_manifest="${TMP_ROOT}/train/${config_name}/runs/atlasmtl/atlasmtl_model_manifest.json"
  [[ -f "${model_manifest}" ]] || { echo "missing model ${model_manifest}" >&2; exit 1; }
  run_dir="${TMP_ROOT}/predict/${config_name}/${point}/runs/atlasmtl"
  mkdir -p "${run_dir}"
  if has_run_outputs "${run_dir}"; then
    echo "=== skip hlca weight predict ${config_name} ${point} ==="
    continue
  fi
  echo "=== hlca weight predict ${config_name} ${point} ==="
  "${PYTHON_BIN}" "${RUNNER}" \
    --dataset-manifest "${manifest_path}" \
    --output-dir "${run_dir}" \
    --atlasmtl-model "${model_manifest}" \
    --device cuda \
    --methods atlasmtl \
    > "${run_dir}/stdout.log" \
    2> "${run_dir}/stderr.log"
done

"${PYTHON_BIN}" "${COLLECT_SCRIPT}"
