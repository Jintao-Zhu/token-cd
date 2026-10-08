#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/leju-suzhou/zjt_ws"
REPO="$ROOT/token-cd"
WORKER="$REPO/scripts/run_univla_l14_worker.sh"
SESSION_PREFIX="univla_l14_simpler_100_20261001"

start_gpu() {
  local gpu="$1"
  local session="${SESSION_PREFIX}_g${gpu}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    return 1
  fi
  tmux new-session -d -s "$session" -n "gpu${gpu}"
  tmux split-window -h -t "$session:0"
  tmux split-window -v -t "$session:0.0"
  tmux split-window -v -t "$session:0.1"
  tmux select-layout -t "$session:0" tiled

  local spoon carrot stack eggplant
  case "$gpu" in
    1) spoon="1-33"; carrot="0-33"; stack="0-33"; eggplant="0-33" ;;
    2) spoon="34-66"; carrot="34-66"; stack="34-66"; eggplant="34-66" ;;
    3) spoon="67-99"; carrot="67-99"; stack="67-99"; eggplant="67-99" ;;
    *) echo "unsupported physical GPU: $gpu" >&2; return 2 ;;
  esac
  tmux send-keys -t "$session:0.0" "sleep 0; $WORKER $gpu widowx_spoon_on_towel $spoon univla_l14_g${gpu}_spoon" C-m
  tmux send-keys -t "$session:0.1" "sleep 8; $WORKER $gpu widowx_carrot_on_plate $carrot univla_l14_g${gpu}_carrot" C-m
  tmux send-keys -t "$session:0.2" "sleep 16; $WORKER $gpu widowx_stack_cube $stack univla_l14_g${gpu}_stack" C-m
  tmux send-keys -t "$session:0.3" "sleep 24; $WORKER $gpu widowx_put_eggplant_in_basket $eggplant univla_l14_g${gpu}_eggplant" C-m
  echo "$session started with four workers on physical GPU $gpu"
}

start_gpu 1
start_gpu 2
start_gpu 3
