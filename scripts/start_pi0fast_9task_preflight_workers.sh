#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD="$ROOT/token-cd"
OUT="$TOKEN_CD/runs/pi0fast_9task_vanilla_vs_l6jit_lambda050_rtfull_100seed_v1"
WORKER_LAUNCHER="$TOKEN_CD/scripts/run_pi0fast_9task_jit_compare_worker.sh"

for spec in "4:18841" "5:18842" "6:18843" "7:18844"; do
  gpu="${spec%%:*}" port="${spec##*:}"
  tmux has-session -t "pi0f9_v05_srv${gpu}" 2>/dev/null || { echo "Missing server session for GPU $gpu" >&2; exit 2; }
  ss -ltn | grep -q ":$port " || { echo "Server port $port is not listening" >&2; exit 2; }
done

start_worker() {
  local gpu="$1" port="$2" family="$3" tag="$4"
  shift 4
  local session="pi0f9_v05_pre_${tag}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Refusing to replace existing tmux session $session." >&2
    exit 2
  fi
  local args=""
  for job in "$@"; do args+=" $job"; done
  tmux new-session -d -s "$session" "bash -lc '$WORKER_LAUNCHER $gpu $port $family $OUT${args}'"
}

start_worker 4 18841 google g4 google_robot_open_drawer:0-1 google_robot_close_drawer:0-1 google_robot_pick_coke_can:0-1
start_worker 5 18842 google g5 google_robot_move_near:0-1 google_robot_place_apple_in_closed_top_drawer:0-1
start_worker 6 18843 widowx w6 widowx_carrot_on_plate:0-1 widowx_put_eggplant_in_basket:0-1
start_worker 7 18844 widowx w7 widowx_spoon_on_towel:0-1 widowx_stack_cube:0-1
echo "Preflight workers launched."
