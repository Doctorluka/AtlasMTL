#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python}"
RUNNER="${REPO_ROOT}/benchmark/pipelines/run_benchmark.py"
MANIFEST_SCRIPT="${SCRIPT_DIR}/generate_phase6c_manifests.py"
COLLECT_SCRIPT="${SCRIPT_DIR}/collect_phase6c_results.py"
TRAIN_INDEX="${REPO_ROOT}/documents/experiments/2026-03-09_phmap_study_split_validation/manifests/phase6c/train_manifest_index.json"
TMP_ROOT="/tmp/atlasmtl_benchmarks/2026-03-10/phmap_study_split_phase6c"

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

"${PYTHON_BIN}" "${MANIFEST_SCRIPT}"

"${PYTHON_BIN}" - "${TRAIN_INDEX}" <<'PY' | while IFS=$'\t' read -r config_name seed manifest_path; do
import json
import sys
rows = json.load(open(sys.argv[1]))
for item in rows:
    print(f"{item['config_name']}\t{item['seed']}\t{item['train_manifest_path']}")
PY
  [[ -n "${manifest_path}" ]] || continue
  run_dir="${TMP_ROOT}/train/${config_name}/seed_${seed}/runs/atlasmtl"
  mkdir -p "${run_dir}"
  if has_run_outputs "${run_dir}"; then
    echo "=== skip phase6c train ${config_name} seed_${seed} ==="
    continue
  fi
  echo "=== phase6c train ${config_name} seed_${seed} ==="
  "${PYTHON_BIN}" "${RUNNER}" \
    --dataset-manifest "${manifest_path}" \
    --output-dir "${run_dir}" \
    --device cuda \
    --methods atlasmtl \
    > "${run_dir}/stdout.log" \
    2> "${run_dir}/stderr.log"
done

"${PYTHON_BIN}" "${COLLECT_SCRIPT}"
