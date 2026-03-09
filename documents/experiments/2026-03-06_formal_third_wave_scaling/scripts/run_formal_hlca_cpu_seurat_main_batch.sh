#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python}"
RUNNER="${REPO_ROOT}/documents/experiments/common/run_reference_heldout_scaleout_benchmark.py"
MANIFEST_DIR="${REPO_ROOT}/documents/experiments/2026-03-06_formal_third_wave_scaling/manifests/reference_heldout"
OUT_ROOT="${OUT_ROOT:-/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/HLCA_Core/benchmark/formal_main/cpu_seurat}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"

BUILD_SIZES=(10000 20000 30000 50000)

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export NUMBA_CACHE_DIR="${REPO_ROOT}/.tmp/numba_cache"
export PYTHONPATH="${REPO_ROOT}"
export ATLASMTL_FAIRNESS_POLICY="cpu_only_strict"

mkdir -p "${REPO_ROOT}/.tmp/numba_cache"
mkdir -p "${OUT_ROOT}"

should_skip_point() {
  local out_dir="$1"
  local status_json="${out_dir}/scaleout_status.json"
  local timeout_log="${out_dir}/manual_stop_long_runtime.json"
  if [[ -f "${timeout_log}" ]]; then
    return 0
  fi
  if [[ ! -f "${status_json}" ]]; then
    return 1
  fi
  "${PYTHON_BIN}" - "${status_json}" <<'PY'
import json
import sys

path = sys.argv[1]
payload = json.load(open(path))
methods = payload.get("methods") or []
ok = bool(methods) and all(m.get("status") == "success" for m in methods)
raise SystemExit(0 if ok else 1)
PY
}

run_with_guardrail() {
  local manifest="$1"
  local out_dir="$2"
  local label="$3"
  local timeout_log="${out_dir}/manual_stop_long_runtime.json"

  mkdir -p "${out_dir}"
  echo "=== HLCA CPU seurat track: ${label} (timeout ${TIMEOUT_SECONDS}s) ==="
  if timeout --signal=TERM --kill-after=30 "${TIMEOUT_SECONDS}" \
    "${PYTHON_BIN}" "${RUNNER}" \
      --dataset-manifest "${manifest}" \
      --output-dir "${out_dir}" \
      --device cpu \
      --methods seurat_anchor_transfer; then
    return 0
  fi

  status=$?
  if [[ "${status}" -eq 124 || "${status}" -eq 137 ]]; then
    cat >"${timeout_log}" <<EOF
{
  "status": "manual_stop_long_runtime",
  "timeout_seconds": ${TIMEOUT_SECONDS},
  "method": "seurat_anchor_transfer",
  "manifest": "${manifest}",
  "output_dir": "${out_dir}"
}
EOF
    echo "manual_stop_long_runtime recorded for ${label}"
    return 0
  fi

  return "${status}"
}

for build_size in "${BUILD_SIZES[@]}"; do
  manifest="${MANIFEST_DIR}/HLCA_Core__ann_level_5__formal_build_scaling_cpu_build${build_size}_eval10k_v1.yaml"
  out_dir="${OUT_ROOT}/build_${build_size}_eval10k"
  if should_skip_point "${out_dir}"; then
    echo "=== skip existing HLCA CPU seurat track: build=${build_size} -> eval10k ==="
    continue
  fi
  run_with_guardrail "${manifest}" "${out_dir}" "build=${build_size} -> eval10k"
done
