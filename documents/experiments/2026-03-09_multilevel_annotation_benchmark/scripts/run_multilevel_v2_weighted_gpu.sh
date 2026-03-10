#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python}"
RUNNER="${REPO_ROOT}/benchmark/pipelines/run_benchmark.py"
INDEX_JSON="${REPO_ROOT}/documents/experiments/2026-03-09_multilevel_annotation_benchmark/manifests/multilevel_v2_weighted_gpu/manifest_index.json"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export NUMBA_CACHE_DIR="${REPO_ROOT}/.tmp/numba_cache"
export PYTHONPATH="${REPO_ROOT}"
export ATLASMTL_FAIRNESS_POLICY="mixed_backend_labeled"

mkdir -p "${REPO_ROOT}/.tmp/numba_cache"

"${PYTHON_BIN}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("GPU preflight failed: torch.cuda.is_available() is false")
print("GPU preflight ok:", torch.cuda.get_device_name(0))
PY

should_skip_point() {
  local out_dir="$1"
  local metrics_json="${out_dir}/runs/atlasmtl/metrics.json"
  local summary_csv="${out_dir}/runs/atlasmtl/summary.csv"
  if [[ ! -f "${metrics_json}" || ! -f "${summary_csv}" ]]; then
    return 1
  fi
  return 0
}

"${PYTHON_BIN}" - "${INDEX_JSON}" <<'PY' | while IFS=$'\t' read -r dataset point manifest; do
import json
import sys

rows = json.load(open(sys.argv[1]))
for item in rows:
    print(f"{item['dataset_name']}\t{item['point']}\t{item['manifest_path']}")
PY
  [[ -n "${manifest}" ]] || continue
  out_dir="/tmp/atlasmtl_benchmarks/2026-03-09/multilevel_annotation_v2_weighted_gpu/${dataset}/benchmark/gpu/${point}"
  mkdir -p "${out_dir}/runs/atlasmtl"
  if should_skip_point "${out_dir}"; then
    echo "=== skip existing multilevel v2 weighted gpu ${dataset} ${point} ==="
    continue
  fi
  echo "=== multilevel v2 weighted gpu ${dataset} ${point} ==="
  "${PYTHON_BIN}" "${RUNNER}" \
    --dataset-manifest "${manifest}" \
    --output-dir "${out_dir}/runs/atlasmtl" \
    --device cuda \
    --methods atlasmtl \
    > "${out_dir}/runs/atlasmtl/stdout.log" \
    2> "${out_dir}/runs/atlasmtl/stderr.log"
done
