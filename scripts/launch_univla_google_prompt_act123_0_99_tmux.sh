#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/leju-suzhou/zjt_ws"
REPO="$ROOT/token-cd"
WORKER="$REPO/scripts/run_univla_google_prompt_act123_worker.sh"

launch() {
  local gpu="$1" task="$2" seeds="$3" depth="$4" name="$5"
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "tmux session already exists: $name" >&2
    return 1
  fi
  tmux new-session -d -s "$name" "bash -lc '$WORKER $gpu $task $seeds $name $depth'"
}

# Nine workers total: three independent UniVLA+SIMPLER processes on each GPU 1-3.
# RT-full = depth 8. RT-depth0 = depth 0. Both hidden-CD lambdas are evaluated
# from the same saved episode snapshot by each rollout process.
launch 1 google_robot_open_drawer 0-49 8 univla_h123_g1_full_open_000_049
launch 1 google_robot_open_drawer 50-99 8 univla_h123_g1_full_open_050_099
launch 1 google_robot_close_drawer 0-99 8 univla_h123_g1_full_close_000_099

launch 2 google_robot_pick_coke_can 0-99 8 univla_h123_g2_full_pick_000_099
launch 2 google_robot_move_near 0-99 8 univla_h123_g2_full_move_000_099
launch 2 google_robot_open_drawer 0-99 0 univla_h123_g2_d0_open_000_099

launch 3 google_robot_close_drawer 0-99 0 univla_h123_g3_d0_close_000_099
launch 3 google_robot_pick_coke_can 0-99 0 univla_h123_g3_d0_pick_000_099
launch 3 google_robot_move_near 0-99 0 univla_h123_g3_d0_move_000_099

tmux list-sessions -F '#{session_name}' | rg '^univla_h123_g[123]_'
