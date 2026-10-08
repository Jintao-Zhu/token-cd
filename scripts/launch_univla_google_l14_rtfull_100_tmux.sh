#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/leju-suzhou/zjt_ws
WORKER="$ROOT/token-cd/scripts/run_univla_l14_google_worker.sh"
SESSION_PREFIX=univla_google_l14_rtfull_20261002
launch_gpu() {
  local gpu="$1" task1="$2" seeds1="$3" task2="$4" seeds2="$5" task3="$6" seeds3="$7"
  local session="${SESSION_PREFIX}_g${gpu}"
  local id1="univla_google_l14_rtfull_g${gpu}_w1_${task1#google_robot_}"
  local id2="univla_google_l14_rtfull_g${gpu}_w2_${task2#google_robot_}"
  local id3="univla_google_l14_rtfull_g${gpu}_w3_${task3#google_robot_}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2; return 1
  fi
  tmux new-session -d -s "$session" -n "gpu${gpu}" "$WORKER $gpu $task1 $seeds1 $id1"
  tmux split-window -t "$session" -v "$WORKER $gpu $task2 $seeds2 $id2"
  tmux split-window -t "$session" -v "$WORKER $gpu $task3 $seeds3 $id3"
  tmux select-layout -t "$session" tiled
}
launch_gpu 1 google_robot_open_drawer 1-25 google_robot_close_drawer 1-25 google_robot_pick_coke_can 1-33
launch_gpu 2 google_robot_open_drawer 26-50 google_robot_close_drawer 26-50 google_robot_pick_coke_can 34-66
launch_gpu 3 google_robot_open_drawer 51-75 google_robot_close_drawer 51-75 google_robot_pick_coke_can 67-99
launch_gpu 4 google_robot_open_drawer 76-99 google_robot_close_drawer 76-99 google_robot_move_near 1-25
launch_gpu 5 google_robot_move_near 26-50 google_robot_move_near 51-75 google_robot_move_near 76-99
launch_gpu 6 google_robot_place_apple_in_closed_top_drawer 1-17 google_robot_place_apple_in_closed_top_drawer 18-33 google_robot_place_apple_in_closed_top_drawer 34-50
launch_gpu 7 google_robot_place_apple_in_closed_top_drawer 51-66 google_robot_place_apple_in_closed_top_drawer 67-83 google_robot_place_apple_in_closed_top_drawer 84-99
