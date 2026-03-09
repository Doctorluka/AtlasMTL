#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python}"
RUNNER="${REPO_ROOT}/documents/experiments/common/run_reference_heldout_scaleout_benchmark.py"
INDEX_JSON="${REPO_ROOT}/documents/experiments/2026-03-06_formal_third_wave_scaling/manifests/reference_heldout/manifest_index.json"
DATASETS=(${DATASETS:-PHMap_Lung_Full_v43_light mTCA DISCO_hPBMCs})
GPU_METHODS=(atlasmtl scanvi)

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

emit_points() {
  local dataset="$1"
  local device_group="$2"
  "${PYTHON_BIN}" - "${INDEX_JSON}" "${dataset}" "${device_group}" <<'PY'
import json
import sys

idx = json.load(open(sys.argv[1]))
dataset = sys.argv[2]
device_group = sys.argv[3]

rows = []
for item in idx:
    if item["dataset_name"] != dataset or item["device_group"] != device_group:
        continue
    if item["track"] == "build_scaling":
        point = f"build_{item['build_size']}_eval10k"
        order = (0, int(item["build_size"]))
    else:
        point = f"predict_{item['build_size']}_{item['predict_size']}"
        order = (1, int(item["predict_size"]))
    rows.append((order, item["manifest_path"], point))

for _, manifest, point in sorted(rows):
    print(f"{manifest}\t{point}")
PY
}

for dataset in "${DATASETS[@]}"; do
  out_root="/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/${dataset}/benchmark/formal_main/gpu"
  mkdir -p "${out_root}"
  while IFS=$'\t' read -r manifest point; do
    [[ -n "${manifest}" ]] || continue
    out_dir="${out_root}/${point}"
    if should_skip_point "${out_dir}"; then
      echo "=== skip existing ${dataset} GPU: ${point} ==="
      continue
    fi
    echo "=== ${dataset} GPU: ${point} ==="
    "${PYTHON_BIN}" "${RUNNER}" \
      --dataset-manifest "${manifest}" \
      --output-dir "${out_dir}" \
      --device cuda \
      --methods "${GPU_METHODS[@]}"
  done < <(emit_points "${dataset}" "gpu")
done
