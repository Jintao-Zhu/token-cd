#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/leju-suzhou/zjt_ws"
REPO="$ROOT/token-cd"
WORKER="$REPO/scripts/run_univla_fractal_google_ft30k_depth0_worker.sh"
start() {
  local name="$1" gpu="$2" task="$3" seeds="$4" worker_id="$5"
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "tmux session already exists: $name" >&2
    exit 1
  fi
  tmux new-session -d -s "$name" "bash '$WORKER' '$gpu' '$task' '$seeds' '$worker_id'"
  echo "started $name on GPU $gpu: $task seeds $seeds"
}
# Match the RT-full shard boundaries exactly: four 0-49/50-99 drawer shards,
# one 0-99 Coke shard, one 0-99 Move Near shard.
start univla_d0_g1_open_000_049 1 google_robot_open_drawer 0-49 univla_d0_g1_open_000_049
start univla_d0_g1_close_000_049 1 google_robot_close_drawer 0-49 univla_d0_g1_close_000_049
start univla_d0_g1_pick_000_099 1 google_robot_pick_coke_can 0-99 univla_d0_g1_pick_000_099
start univla_d0_g6_open_050_099 6 google_robot_open_drawer 50-99 univla_d0_g6_open_050_099
start univla_d0_g6_close_050_099 6 google_robot_close_drawer 50-99 univla_d0_g6_close_050_099
start univla_d0_g6_move_000_099 6 google_robot_move_near 0-99 univla_d0_g6_move_000_099
