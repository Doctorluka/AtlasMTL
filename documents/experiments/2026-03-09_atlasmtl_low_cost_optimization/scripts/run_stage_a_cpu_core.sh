#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python}"
RUNNER="${REPO_ROOT}/documents/experiments/common/run_reference_heldout_scaleout_benchmark.py"
INDEX_JSON="${REPO_ROOT}/documents/experiments/2026-03-09_atlasmtl_low_cost_optimization/manifests/stage_a/manifest_index.json"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export NUMBA_CACHE_DIR="${REPO_ROOT}/.tmp/numba_cache"
export PYTHONPATH="${REPO_ROOT}"
export ATLASMTL_FAIRNESS_POLICY="cpu_only_strict"

mkdir -p "${REPO_ROOT}/.tmp/numba_cache"

"${PYTHON_BIN}" - "${INDEX_JSON}" <<'PY' | while IFS=$'\t' read -r dataset point config manifest; do
import json
import sys

rows = json.load(open(sys.argv[1]))
for row in rows:
    if row["device_group"] != "cpu":
        continue
    print(f'{row["dataset_name"]}\t{row["point"]}\t{row["config_name"]}\t{row["manifest_path"]}')
PY
  [[ -n "${manifest}" ]] || continue
  out_dir="/tmp/atlasmtl_benchmarks/2026-03-09/atlasmtl_low_cost_optimization/${dataset}/benchmark/stage_a/cpu_core/${point}/${config}"
  mkdir -p "${out_dir}"
  echo "=== stage_a cpu_core ${dataset} ${point} ${config} ==="
  "${PYTHON_BIN}" "${RUNNER}" \
    --dataset-manifest "${manifest}" \
    --output-dir "${out_dir}" \
    --device cpu \
    --methods atlasmtl
done
