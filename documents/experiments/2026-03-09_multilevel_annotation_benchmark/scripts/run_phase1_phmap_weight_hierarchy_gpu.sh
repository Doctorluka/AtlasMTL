#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python}"
RUNNER="${REPO_ROOT}/benchmark/pipelines/run_benchmark.py"
MANIFEST_SCRIPT="${SCRIPT_DIR}/generate_phase1_phmap_weight_hierarchy_manifests.py"
TRAIN_INDEX="${REPO_ROOT}/documents/experiments/2026-03-09_multilevel_annotation_benchmark/manifests/phase1_phmap_weight_hierarchy/train_manifest_index.json"
PREDICT_INDEX="${REPO_ROOT}/documents/experiments/2026-03-09_multilevel_annotation_benchmark/manifests/phase1_phmap_weight_hierarchy/predict_manifest_index.json"
TMP_ROOT="/tmp/atlasmtl_benchmarks/2026-03-09/multilevel_annotation_phase1_phmap_weight_hierarchy"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export NUMBA_CACHE_DIR="${REPO_ROOT}/.tmp/numba_cache"
export PYTHONPATH="${REPO_ROOT}"
export ATLASMTL_FAIRNESS_POLICY="mixed_backend_labeled"

mkdir -p "${REPO_ROOT}/.tmp/numba_cache"

"${PYTHON_BIN}" "${MANIFEST_SCRIPT}"

"${PYTHON_BIN}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("GPU preflight failed: torch.cuda.is_available() is false")
print("GPU preflight ok:", torch.cuda.get_device_name(0))
PY

has_run_outputs() {
  local out_dir="$1"
  local metrics_json="${out_dir}/metrics.json"
  local summary_csv="${out_dir}/summary.csv"
  if [[ ! -f "${metrics_json}" || ! -f "${summary_csv}" ]]; then
    return 1
  fi
  return 0
}

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
    echo "=== skip existing phase1 train ${config_name} ==="
    continue
  fi
  echo "=== phase1 train ${config_name} ==="
  "${PYTHON_BIN}" "${RUNNER}" \
    --dataset-manifest "${manifest_path}" \
    --output-dir "${run_dir}" \
    --device cuda \
    --methods atlasmtl \
    > "${run_dir}/stdout.log" \
    2> "${run_dir}/stderr.log"
done

"${PYTHON_BIN}" - "${PREDICT_INDEX}" <<'PY' | while IFS=$'\t' read -r config_name point hierarchy_setting manifest_path; do
import json
import sys

rows = json.load(open(sys.argv[1]))
for item in rows:
    print(
        f"{item['config_name']}\t{item['point']}\t{item['hierarchy_setting']}\t{item['predict_manifest_path']}"
    )
PY
  [[ -n "${manifest_path}" ]] || continue
  model_manifest="${TMP_ROOT}/train/${config_name}/runs/atlasmtl/atlasmtl_model_manifest.json"
  if [[ ! -f "${model_manifest}" ]]; then
    echo "missing trained model for ${config_name}: ${model_manifest}" >&2
    exit 1
  fi
  run_dir="${TMP_ROOT}/predict/${config_name}/${point}/hierarchy_${hierarchy_setting}/runs/atlasmtl"
  mkdir -p "${run_dir}"
  if has_run_outputs "${run_dir}"; then
    echo "=== skip existing phase1 predict ${config_name} ${point} hierarchy_${hierarchy_setting} ==="
    continue
  fi
  echo "=== phase1 predict ${config_name} ${point} hierarchy_${hierarchy_setting} ==="
  "${PYTHON_BIN}" "${RUNNER}" \
    --dataset-manifest "${manifest_path}" \
    --output-dir "${run_dir}" \
    --atlasmtl-model "${model_manifest}" \
    --device cuda \
    --methods atlasmtl \
    > "${run_dir}/stdout.log" \
    2> "${run_dir}/stderr.log"
done
