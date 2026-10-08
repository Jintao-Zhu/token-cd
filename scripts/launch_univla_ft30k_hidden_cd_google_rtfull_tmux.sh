#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/leju-suzhou/zjt_ws"
REPO="$ROOT/token-cd"
WORKER="$REPO/scripts/run_univla_ft30k_hidden_cd_google_worker.sh"

launch() {
  local gpu="$1" task="$2" seeds="$3" name="$4"
  case "$gpu" in 3|4) echo "GPU $gpu is explicitly excluded" >&2; return 2;; esac
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "tmux session already exists: $name" >&2
    return 1
  fi
  tmux new-session -d -s "$name" "bash -lc '$WORKER $gpu $task $seeds $name'"
}

# Eight independent workers, balanced on 0/1/6/7. GPUs 3 and 4 are not used.
launch 0 google_robot_open_drawer 0-49 univla_hcd30k_g0_open_000_049
launch 0 google_robot_close_drawer 0-49 univla_hcd30k_g0_close_000_049
launch 1 google_robot_pick_coke_can 0-49 univla_hcd30k_g1_pick_000_049
launch 1 google_robot_move_near 0-49 univla_hcd30k_g1_move_000_049
launch 6 google_robot_open_drawer 50-99 univla_hcd30k_g6_open_050_099
launch 6 google_robot_close_drawer 50-99 univla_hcd30k_g6_close_050_099
launch 7 google_robot_pick_coke_can 50-99 univla_hcd30k_g7_pick_050_099
launch 7 google_robot_move_near 50-99 univla_hcd30k_g7_move_050_099

tmux list-sessions -F '#{session_name}' | rg '^univla_hcd30k_g(0|1|6|7)_'
