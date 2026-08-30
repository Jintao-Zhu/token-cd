#!/usr/bin/env bash
# Sequential orchestrator for the 6h compute budget. Three phases, each launching
# as many shards as fit in 24 procs (4/GPU x 6 GPUs), then waiting for DONE markers
# before starting the next phase. Ends with full statistics.
set -u
cd ~/zjt_ws/token-cd

PY=~/zjt_ws/openvla-ar-h100/bin/python
SCD=research/semantic_token_cd
WS=artifacts/attn_semantic_layer_window_sweep_v1
LS=artifacts/attn_semantic_lambda_sweep_v1
SS=artifacts/attn_semantic_selection_sweep_v1

# Wait until `expected` log files under `glob` contain a DONE line, with a 30-min
# stall detector and a hard timeout.
wait_done() {
  local glob=$1 expected=$2 timeout_sec=${3:-14400}
  local waited=0 stall=0 last=-1 got=0
  while [ $waited -lt $timeout_sec ]; do
    got=$(grep -ls '"DONE": true' $glob 2>/dev/null | wc -l)
    if [ "$got" -ge "$expected" ]; then
      echo "[$(date +%T)] DONE $got/$expected: $glob"; return 0
    fi
    if [ "$got" -eq "$last" ]; then stall=$((stall+30)); else stall=0; last=$got; fi
    if [ $stall -ge 1800 ]; then
      echo "[$(date +%T)] WARN stalled ${got}/${expected} (30min): $glob"; return 1
    fi
    sleep 30; waited=$((waited+30))
  done
  got=$(grep -ls '"DONE": true' $glob 2>/dev/null | wc -l)
  echo "[$(date +%T)] WARN timeout ${got}/${expected}: $glob"; return 1
}

echo "[$(date +%T)] === Phase 1: layer-window sweep (3 tasks, 24 procs) ==="
bash "$SCD/launch_layer_window_sweep.sh"
wait_done "$WS/logs/*.log" 24

echo "[$(date +%T)] === Phase 2: lambda + selection sweep (24 procs) ==="
bash "$SCD/launch_lambda_sweep.sh"
bash "$SCD/launch_selection_sweep.sh"
wait_done "$LS/logs/*.log" 8
wait_done "$SS/logs/*.log" 16

echo "[$(date +%T)] === Phase 3: window sweep extension, open_drawer + stack_cube (16 procs) ==="
bash "$SCD/launch_window_sweep_extended.sh"
wait_done "$WS/logs/*.log" 40

echo "[$(date +%T)] === Statistics ==="
echo "----- layer window sweep -----"
"$PY" "$SCD/analyze_layer_window_sweep.py" --artifact "$WS"
echo "----- lambda sweep -----"
"$PY" "$SCD/analyze_lambda_sweep.py" --artifact "$LS"
echo "----- selection sweep -----"
"$PY" "$SCD/analyze_selection_sweep.py" --artifact "$SS"
echo "[$(date +%T)] ALL PHASES COMPLETE"
