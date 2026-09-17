#!/usr/bin/env bash
set -euo pipefail

WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ROOT="$WS/artifacts/l11_matched_budget_provenance_offline_v2"
mkdir -p "$ROOT/logs"
cd "$WS"
export PYTHONPATH="$WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"

run_task() {
  local gpu="$1" task="$2"
  "$PY" research/semantic_token_cd/offline_l11_budget_provenance_worker.py \
    --artifact "$ROOT" --task "$task" --gpu "$gpu" \
    >"$ROOT/logs/${task}.log" 2>&1
}

run_task 2 google_robot_open_drawer & p1=$!
run_task 3 google_robot_close_drawer & p2=$!
wait "$p1" "$p2"
run_task 2 google_robot_pick_coke_can & p3=$!
run_task 3 google_robot_move_near & p4=$!
wait "$p3" "$p4"

"$PY" research/semantic_token_cd/analyze_l11_budget_provenance_offline.py \
  --artifact "$ROOT" >"$ROOT/logs/analysis.log" 2>&1
