#!/usr/bin/env zsh
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <job-script> <log-file> <pid-file>" >&2
  exit 1
fi

JOB_SCRIPT="$1"
LOG_FILE="$2"
PID_FILE="$3"

mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$PID_FILE")"

nohup "$JOB_SCRIPT" >"$LOG_FILE" 2>&1 < /dev/null &
JOB_PID=$!
echo "$JOB_PID" >"$PID_FILE"
echo "started:${JOB_PID}"
