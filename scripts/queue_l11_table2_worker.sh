#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 5 ]]; then
  echo "usage: $0 GPU CARD ARM WORKER_ID LOG" >&2
  exit 2
fi
GPU=$1 CARD=$2 ARM=$3 WORKER=$4 LOG=$5
ROOT=/home/leju-suzhou/zjt_ws/token-cd
LOCK="/tmp/l11_table2_gpu${GPU}.launch.lock"
count_workers() {
  ps -eo args= | awk -v gpu="$GPU" '
    !/awk/ && /python research\/semantic_token_cd\/l11_table2_openvla_rollout.py/ &&
    $0 ~ ("--gpu " gpu " ") { n++ }
    END { print n+0 }
  '
}
while true; do
  exec 9>"$LOCK"
  flock -x 9
  active=$(count_workers)
  if (( active < 3 )); then
    bash "$ROOT/scripts/run_l11_table2_worker.sh" "$GPU" "$CARD" "$ARM" "$WORKER" >"$LOG" 2>&1 &
    child=$!
    for _ in $(seq 1 90); do
      now=$(count_workers)
      if (( now > active )); then
        flock -u 9
        wait "$child"
        exit $?
      fi
      if ! kill -0 "$child" 2>/dev/null; then
        wait "$child"
        exit $?
      fi
      sleep 2
    done
    kill "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  flock -u 9
  sleep 15
done
