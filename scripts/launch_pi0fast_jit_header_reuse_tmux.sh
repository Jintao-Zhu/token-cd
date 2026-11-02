#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD="$ROOT/token-cd"
OUT="$TOKEN_CD/runs/pi0fast_jit_header_google4_100seed_rtfull_v1"
mkdir -p "$OUT/logs"

GPU_IDS=(1 2 3 4)
TASKS=(google_robot_open_drawer google_robot_close_drawer google_robot_pick_coke_can google_robot_move_near)

for i in 0 1 2 3; do
  for shard in 0 1 2; do
    port=$((18761+i+100*shard))
    if [[ "$shard" -eq 0 ]]; then sess="pi0hdr100_srv_${i}"; else sess="pi0hdr100_srv_${i}_${shard}"; fi
    if ! tmux has-session -t "$sess" 2>/dev/null; then
      tmux new-session -d -s "$sess" \
        "env PI0FAST_ARTIFACT_DIR=$OUT PI0FAST_TIMING=1 PI0FAST_CD_JIT_HEADER=1 bash $TOKEN_CD/scripts/run_pi0fast_l6_server.sh ${GPU_IDS[$i]} $port jit100_${shard}"
    fi
  done
done

for i in 0 1 2 3; do
 for shard in 0 1 2; do
  port=$((18761+i+100*shard))
  ready=0
  for attempt in $(seq 1 120); do
    if python3 - "$port" <<'PY'
import socket, sys
try:
    with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1):
        pass
except OSError:
    raise SystemExit(1)
PY
    then
      ready=1
      break
    fi
    sleep 2
  done
  if [[ "$ready" != 1 ]]; then
    echo "server on port $port did not become ready" >&2
    exit 1
  fi
 done
done

# One independent policy service and renderer client per seed shard.
for i in 0 1 2 3; do
  for shard in 0 1 2; do
    case "$shard" in
      0) start=0; end=33 ;;
      1) start=34; end=66 ;;
      2) start=67; end=99 ;;
    esac
    sess="pi0hdr100_w_${i}_${shard}"
    if ! tmux has-session -t "$sess" 2>/dev/null; then
      port=$((18761+i+100*shard))
      tmux new-session -d -s "$sess" \
        "bash $TOKEN_CD/scripts/run_pi0fast_jit_header_reuse_worker.sh ${GPU_IDS[$i]} ${TASKS[$i]} $port $start $end ${i}_${shard}"
    fi
  done
done

if ! tmux has-session -t pi0hdr100_monitor 2>/dev/null; then
  tmux new-session -d -s pi0hdr100_monitor \
    "bash -lc 'cd $TOKEN_CD && python3 -u scripts/monitor_pi0fast_jit_reuse.py >> $OUT/logs/monitor.log 2>&1'"
fi
