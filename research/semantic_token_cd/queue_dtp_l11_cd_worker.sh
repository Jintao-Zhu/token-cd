#!/usr/bin/env bash
set -euo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
ROOT="$WS/artifacts/dtp_fixed_positive_l11_matched_cd_v1"
slot="${1:?worker slot required}"
gpu="${slot:1:1}"
mkdir -p "$ROOT/logs"
# Each queue waits for the corresponding Full-Joint tmux session itself to
# disappear.  This avoids a race during the short gap between two Python tasks
# launched sequentially by the old worker shell.
while tmux has-session -t "pa_full_$slot" 2>/dev/null; do sleep 20; done
while [ ! -f "$ROOT/preflight/PASS" ]; do sleep 10; done
# The rollout is file-resumable.  Restart the shard after a transient process
# failure; already complete summary/array/video triples are skipped.
attempt=0
while true; do
  if bash "$WS/research/semantic_token_cd/run_dtp_l11_cd1200.sh" "$slot"; then
    exit 0
  fi
  attempt=$((attempt + 1))
  echo "[$(date --iso-8601=seconds)] shard=$slot retry=$attempt" >&2
  sleep 30
done
