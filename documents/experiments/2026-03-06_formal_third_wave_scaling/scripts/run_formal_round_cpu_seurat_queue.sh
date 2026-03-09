#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python}"
RUNNER="${REPO_ROOT}/documents/experiments/common/run_reference_heldout_scaleout_benchmark.py"
INDEX_JSON="${REPO_ROOT}/documents/experiments/2026-03-06_formal_third_wave_scaling/manifests/reference_heldout/manifest_index.json"
DATASETS=(${DATASETS:-PHMap_Lung_Full_v43_light mTCA DISCO_hPBMCs})
CPU_SEURAT_PREDICT_DATASETS=(${CPU_SEURAT_PREDICT_DATASETS:-Vento})

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export NUMBA_CACHE_DIR="${REPO_ROOT}/.tmp/numba_cache"
export PYTHONPATH="${REPO_ROOT}"
export ATLASMTL_FAIRNESS_POLICY="cpu_only_strict"

mkdir -p "${REPO_ROOT}/.tmp/numba_cache"

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

payload = json.load(open(sys.argv[1]))
methods = payload.get("methods") or []
raise SystemExit(0 if methods and all(m.get("status") == "success" for m in methods) else 1)
PY
}

emit_build_points() {
  local dataset="$1"
  "${PYTHON_BIN}" - "${INDEX_JSON}" "${dataset}" <<'PY'
import json
import sys

idx = json.load(open(sys.argv[1]))
dataset = sys.argv[2]
keep = {10000, 20000, 30000, 50000}
rows = []
for item in idx:
    if item["dataset_name"] != dataset or item["device_group"] != "cpu" or item["track"] != "build_scaling":
        continue
    build = int(item["build_size"])
    if build not in keep:
        continue
    point = f"build_{build}_eval10k"
    rows.append((build, item["manifest_path"], point))
for _, manifest, point in sorted(rows):
    print(f"{manifest}\t{point}")
PY
}

emit_predict_points() {
  local dataset="$1"
  "${PYTHON_BIN}" - "${INDEX_JSON}" "${dataset}" <<'PY'
import json
import sys

idx = json.load(open(sys.argv[1]))
dataset = sys.argv[2]
rows = []
for item in idx:
    if item["dataset_name"] != dataset or item["device_group"] != "cpu" or item["track"] != "predict_scaling":
        continue
    point = f"predict_{int(item['build_size'])}_{int(item['predict_size'])}"
    rows.append((int(item["predict_size"]), item["manifest_path"], point))
for _, manifest, point in sorted(rows):
    print(f"{manifest}\t{point}")
PY
}

run_with_guardrail() {
  local manifest="$1"
  local out_dir="$2"
  local label="$3"
  mkdir -p "${out_dir}"
  echo "=== CPU seurat: ${label} ==="
  "${PYTHON_BIN}" "${RUNNER}" \
    --dataset-manifest "${manifest}" \
    --output-dir "${out_dir}" \
    --device cpu \
    --methods seurat_anchor_transfer
}

allow_predict_for_dataset() {
  local dataset="$1"
  local allowed
  for allowed in "${CPU_SEURAT_PREDICT_DATASETS[@]}"; do
    if [[ "${dataset}" == "${allowed}" ]]; then
      return 0
    fi
  done
  return 1
}

for dataset in "${DATASETS[@]}"; do
  out_root="/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/${dataset}/benchmark/formal_main/cpu_seurat"
  mkdir -p "${out_root}"
  while IFS=$'\t' read -r manifest point; do
    [[ -n "${manifest}" ]] || continue
    out_dir="${out_root}/${point}"
    if should_skip_point "${out_dir}"; then
      echo "=== skip existing ${dataset} CPU seurat: ${point} ==="
      continue
    fi
    run_with_guardrail "${manifest}" "${out_dir}" "${dataset} ${point}"
  done < <(emit_build_points "${dataset}")
  if ! allow_predict_for_dataset "${dataset}"; then
    echo "=== skip CPU seurat predict scaling for ${dataset}: exploratory-only outside retained dataset set ==="
    continue
  fi
  while IFS=$'\t' read -r manifest point; do
    [[ -n "${manifest}" ]] || continue
    out_dir="${out_root}/${point}"
    if should_skip_point "${out_dir}"; then
      echo "=== skip existing ${dataset} CPU seurat: ${point} ==="
      continue
    fi
    run_with_guardrail "${manifest}" "${out_dir}" "${dataset} ${point}"
  done < <(emit_predict_points "${dataset}")
done
