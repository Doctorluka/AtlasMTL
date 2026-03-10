#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python}"
RUNNER="${REPO_ROOT}/benchmark/pipelines/run_benchmark.py"

SCREEN_MANIFEST_SCRIPT="${SCRIPT_DIR}/generate_phase2_screen_manifests.py"
SCREEN_COLLECT_SCRIPT="${SCRIPT_DIR}/collect_phase2_screen_results.py"
SCREEN_TRAIN_INDEX="${REPO_ROOT}/documents/experiments/2026-03-09_phmap_study_split_validation/manifests/phase2_screen/train_manifest_index.json"
SCREEN_PREDICT_INDEX="${REPO_ROOT}/documents/experiments/2026-03-09_phmap_study_split_validation/manifests/phase2_screen/predict_manifest_index.json"
SCREEN_TMP_ROOT="/tmp/atlasmtl_benchmarks/2026-03-09/phmap_study_split_phase2_screen"

SEED_MANIFEST_SCRIPT="${SCRIPT_DIR}/generate_phase2_seed_manifests.py"
SEED_COLLECT_SCRIPT="${SCRIPT_DIR}/collect_phase2_seed_results.py"
SEED_TRAIN_INDEX="${REPO_ROOT}/documents/experiments/2026-03-09_phmap_study_split_validation/manifests/phase2_seed/train_manifest_index.json"
SEED_PREDICT_INDEX="${REPO_ROOT}/documents/experiments/2026-03-09_phmap_study_split_validation/manifests/phase2_seed/predict_manifest_index.json"
SEED_TMP_ROOT="/tmp/atlasmtl_benchmarks/2026-03-09/phmap_study_split_phase2_seed"

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
  local metrics_json="${out_dir}/metrics.json"
  local summary_csv="${out_dir}/summary.csv"
  if [[ ! -f "${metrics_json}" || ! -f "${summary_csv}" ]]; then
    return 1
  fi
  return 0
}

gpu_preflight() {
  "${PYTHON_BIN}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("GPU preflight failed: torch.cuda.is_available() is false")
print("GPU preflight ok:", torch.cuda.get_device_name(0))
PY
}

run_screen() {
  "${PYTHON_BIN}" "${SCREEN_MANIFEST_SCRIPT}"

  "${PYTHON_BIN}" - "${SCREEN_TRAIN_INDEX}" <<'PY' | while IFS=$'\t' read -r config_name manifest_path; do
import json
import sys
rows = json.load(open(sys.argv[1]))
for item in rows:
    print(f"{item['config_name']}\t{item['train_manifest_path']}")
PY
    [[ -n "${manifest_path}" ]] || continue
    run_dir="${SCREEN_TMP_ROOT}/train/${config_name}/runs/atlasmtl"
    mkdir -p "${run_dir}"
    if has_run_outputs "${run_dir}"; then
      echo "=== skip phase2 screen train ${config_name} ==="
      continue
    fi
    echo "=== phase2 screen train ${config_name} ==="
    "${PYTHON_BIN}" "${RUNNER}" \
      --dataset-manifest "${manifest_path}" \
      --output-dir "${run_dir}" \
      --device cuda \
      --methods atlasmtl \
      > "${run_dir}/stdout.log" \
      2> "${run_dir}/stderr.log"
  done

  "${PYTHON_BIN}" - "${SCREEN_PREDICT_INDEX}" <<'PY' | while IFS=$'\t' read -r config_name point manifest_path; do
import json
import sys
rows = json.load(open(sys.argv[1]))
for item in rows:
    print(f"{item['config_name']}\t{item['point']}\t{item['predict_manifest_path']}")
PY
    [[ -n "${manifest_path}" ]] || continue
    model_manifest="${SCREEN_TMP_ROOT}/train/${config_name}/runs/atlasmtl/atlasmtl_model_manifest.json"
    [[ -f "${model_manifest}" ]] || { echo "missing model ${model_manifest}" >&2; exit 1; }
    run_dir="${SCREEN_TMP_ROOT}/predict/${config_name}/${point}/runs/atlasmtl"
    mkdir -p "${run_dir}"
    if has_run_outputs "${run_dir}"; then
      echo "=== skip phase2 screen predict ${config_name} ${point} ==="
      continue
    fi
    echo "=== phase2 screen predict ${config_name} ${point} ==="
    "${PYTHON_BIN}" "${RUNNER}" \
      --dataset-manifest "${manifest_path}" \
      --output-dir "${run_dir}" \
      --atlasmtl-model "${model_manifest}" \
      --device cuda \
      --methods atlasmtl \
      > "${run_dir}/stdout.log" \
      2> "${run_dir}/stderr.log"
  done

  "${PYTHON_BIN}" "${SCREEN_COLLECT_SCRIPT}"
}

run_seed() {
  "${PYTHON_BIN}" "${SEED_MANIFEST_SCRIPT}"

  "${PYTHON_BIN}" - "${SEED_TRAIN_INDEX}" <<'PY' | while IFS=$'\t' read -r config_name seed manifest_path; do
import json
import sys
rows = json.load(open(sys.argv[1]))
for item in rows:
    print(f"{item['config_name']}\t{item['seed']}\t{item['train_manifest_path']}")
PY
    [[ -n "${manifest_path}" ]] || continue
    run_dir="${SEED_TMP_ROOT}/train/${config_name}/seed_${seed}/runs/atlasmtl"
    mkdir -p "${run_dir}"
    if has_run_outputs "${run_dir}"; then
      echo "=== skip phase2 seed train ${config_name} seed_${seed} ==="
      continue
    fi
    echo "=== phase2 seed train ${config_name} seed_${seed} ==="
    "${PYTHON_BIN}" "${RUNNER}" \
      --dataset-manifest "${manifest_path}" \
      --output-dir "${run_dir}" \
      --device cuda \
      --methods atlasmtl \
      > "${run_dir}/stdout.log" \
      2> "${run_dir}/stderr.log"
  done

  "${PYTHON_BIN}" - "${SEED_PREDICT_INDEX}" <<'PY' | while IFS=$'\t' read -r config_name seed point manifest_path; do
import json
import sys
rows = json.load(open(sys.argv[1]))
for item in rows:
    print(f"{item['config_name']}\t{item['seed']}\t{item['point']}\t{item['predict_manifest_path']}")
PY
    [[ -n "${manifest_path}" ]] || continue
    model_manifest="${SEED_TMP_ROOT}/train/${config_name}/seed_${seed}/runs/atlasmtl/atlasmtl_model_manifest.json"
    [[ -f "${model_manifest}" ]] || { echo "missing model ${model_manifest}" >&2; exit 1; }
    run_dir="${SEED_TMP_ROOT}/predict/${config_name}/seed_${seed}/${point}/runs/atlasmtl"
    mkdir -p "${run_dir}"
    if has_run_outputs "${run_dir}"; then
      echo "=== skip phase2 seed predict ${config_name} seed_${seed} ${point} ==="
      continue
    fi
    echo "=== phase2 seed predict ${config_name} seed_${seed} ${point} ==="
    "${PYTHON_BIN}" "${RUNNER}" \
      --dataset-manifest "${manifest_path}" \
      --output-dir "${run_dir}" \
      --atlasmtl-model "${model_manifest}" \
      --device cuda \
      --methods atlasmtl \
      > "${run_dir}/stdout.log" \
      2> "${run_dir}/stderr.log"
  done

  "${PYTHON_BIN}" "${SEED_COLLECT_SCRIPT}"
}

gpu_preflight
run_screen
run_seed
