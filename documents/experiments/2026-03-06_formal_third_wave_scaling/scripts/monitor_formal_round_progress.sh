#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ROOT="${ROOT:-/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave}"
LOG_FILE="${LOG_FILE:-${ROOT}/logs/formal_round_progress_monitor.log}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
DATASETS=(${DATASETS:-HLCA_Core PHMap_Lung_Full_v43_light mTCA DISCO_hPBMCs Vento})
TRACKS=(cpu_core gpu cpu_seurat)

mkdir -p "$(dirname "${LOG_FILE}")"

log_line() {
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "${LOG_FILE}"
}

count_json() {
  local base="$1"
  local name="$2"
  find "${base}" -maxdepth 2 -name "${name}" 2>/dev/null | wc -l | tr -d ' '
}

while true; do
  log_line "--- monitor tick start ---"
  for dataset in "${DATASETS[@]}"; do
    for track in "${TRACKS[@]}"; do
      base="${ROOT}/${dataset}/benchmark/formal_main/${track}"
      completed="$(count_json "${base}" 'scaleout_status.json')"
      timeouts="$(count_json "${base}" 'manual_stop_long_runtime.json')"
      log_line "dataset=${dataset} track=${track} completed=${completed} timeouts=${timeouts}"
    done
  done
  ps_snapshot="$(ps -ef | rg 'run_formal_round_(cpu_core|gpu|cpu_seurat)_queue|run_reference_heldout_scaleout_benchmark.py --dataset-manifest .*formal_(build|predict)_scaling|run_seurat_anchor_transfer.R|scanvi' || true)"
  if [[ -n "${ps_snapshot}" ]]; then
    while IFS= read -r line; do
      log_line "ps=${line}"
    done <<< "${ps_snapshot}"
  else
    log_line "ps=none"
  fi
  log_line "--- monitor tick end ---"
  sleep "${INTERVAL_SECONDS}"
done
