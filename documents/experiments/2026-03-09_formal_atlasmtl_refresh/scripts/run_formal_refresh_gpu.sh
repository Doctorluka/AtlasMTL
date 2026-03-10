#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python}"
RUNNER="${REPO_ROOT}/documents/experiments/common/run_reference_heldout_scaleout_benchmark.py"
INDEX_JSON="${REPO_ROOT}/documents/experiments/2026-03-09_formal_atlasmtl_refresh/manifests/refresh/manifest_index.json"

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
  local status_json="${out_dir}/scaleout_status.json"
  local metrics_json="${out_dir}/runs/atlasmtl/metrics.json"
  local summary_csv="${out_dir}/runs/atlasmtl/summary.csv"
  if [[ ! -f "${status_json}" || ! -f "${metrics_json}" || ! -f "${summary_csv}" ]]; then
    return 1
  fi
  "${PYTHON_BIN}" - "${status_json}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
methods = payload.get("methods") or []
atlas = next((item for item in methods if item.get("method") == "atlasmtl"), None)
raise SystemExit(0 if atlas and atlas.get("status") == "success" else 1)
PY
}

"${PYTHON_BIN}" - "${INDEX_JSON}" <<'PY' | while IFS=$'\t' read -r dataset point manifest; do
import json
import sys

rows = json.load(open(sys.argv[1]))
for item in rows:
    if item["device_group"] == "gpu":
        print(f"{item['dataset_name']}\t{item['point']}\t{item['manifest_path']}")
PY
  [[ -n "${manifest}" ]] || continue
  out_dir="/tmp/atlasmtl_benchmarks/2026-03-09/formal_atlasmtl_refresh/${dataset}/benchmark/gpu/${point}"
  mkdir -p "${out_dir}"
  if should_skip_point "${out_dir}"; then
    echo "=== skip existing formal refresh gpu ${dataset} ${point} ==="
    continue
  fi
  echo "=== formal refresh gpu ${dataset} ${point} ==="
  "${PYTHON_BIN}" "${RUNNER}" \
    --dataset-manifest "${manifest}" \
    --output-dir "${out_dir}" \
    --device cuda \
    --methods atlasmtl
done
