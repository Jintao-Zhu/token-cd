#!/usr/bin/env bash
# Wait for the 24 Phase-1 window-sweep shards to finish, then run stats.
set -u
cd ~/zjt_ws/token-cd

WS=artifacts/attn_semantic_layer_window_sweep_v1
PY=~/zjt_ws/openvla-ar-h100/bin/python
SCD=research/semantic_token_cd

wait_done() {
  local glob=$1 expected=$2
  local got=0 last=-1 stall=0 alive=0
  while true; do
    got=$(grep -ls '"DONE": true' $glob 2>/dev/null | wc -l)
    if [ "$got" -ge "$expected" ]; then
      echo "[$(date +%T)] Phase 1 complete: $got/$expected shards DONE"
      return 0
    fi
    alive=$(pgrep -f "attn_semantic_layer_window_sweep.py" | wc -l)
    if [ "$alive" -eq 0 ]; then
      echo "[$(date +%T)] ERROR: no sweep procs alive at $got/$expected (crash?)"
      return 1
    fi
    if [ "$got" -eq "$last" ]; then stall=$((stall+60)); else stall=0; last=$got; fi
    if [ "$stall" -ge 3600 ]; then
      echo "[$(date +%T)] WARN: no progress for 60min ($got/$expected, $alive procs alive)"
      stall=0
    fi
    sleep 60
  done
}

wait_done "$WS/logs/*.log" 24
echo "[$(date +%T)] Running statistics..."
"$PY" "$SCD/analyze_layer_window_sweep.py" --artifact "$WS" | tee "$WS/RESULTS_$(date +%F_%H%M%S).txt"
echo "[$(date +%T)] DONE"
