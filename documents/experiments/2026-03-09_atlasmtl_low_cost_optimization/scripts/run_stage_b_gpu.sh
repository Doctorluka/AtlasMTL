#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python}"
RUNNER="${REPO_ROOT}/documents/experiments/common/run_reference_heldout_scaleout_benchmark.py"
INDEX_JSON="${REPO_ROOT}/documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/manifests/stage_b/manifest_index.json"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export NUMBA_CACHE_DIR="${REPO_ROOT}/.tmp/numba_cache"
export PYTHONPATH="${REPO_ROOT}"
export ATLASMTL_FAIRNESS_POLICY="mixed_backend_labeled"

mkdir -p "${REPO_ROOT}/.tmp/numba_cache"

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

"${PYTHON_BIN}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("GPU preflight failed: torch.cuda.is_available() is false")
print("GPU preflight ok:", torch.cuda.get_device_name(0))
PY

"${PYTHON_BIN}" - "${INDEX_JSON}" <<'PY' | while IFS=$'\t' read -r dataset point config manifest; do
import json
import sys

rows = json.load(open(sys.argv[1]))
for row in rows:
    if row["device_group"] != "gpu":
        continue
    print(f'{row["dataset_name"]}\t{row["point"]}\t{row["config_name"]}\t{row["manifest_path"]}')
PY
  [[ -n "${manifest}" ]] || continue
  out_dir="/tmp/atlasmtl_benchmarks/2026-03-09/atlasmtl_low_cost_optimization/${dataset}/benchmark/stage_b/gpu/${point}/${config}"
  mkdir -p "${out_dir}"
  if should_skip_point "${out_dir}"; then
    echo "=== skip existing stage_b gpu ${dataset} ${point} ${config} ==="
    continue
  fi
  echo "=== stage_b gpu ${dataset} ${point} ${config} ==="
  "${PYTHON_BIN}" "${RUNNER}" \
    --dataset-manifest "${manifest}" \
    --output-dir "${out_dir}" \
    --device cuda \
    --methods atlasmtl
done
