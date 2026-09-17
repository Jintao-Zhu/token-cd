#!/usr/bin/env bash
# VLA-Pruner reproduction -- 22-scene closed-loop calibration on GPUs 2/3.
# Two workers per GPU; every worker restores the same canonical snapshot per
# (task, seed, arm) so Rescue/Harm is exactly paired.
set -u
WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ENV="export PYTHONPATH=$WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source; export HF_HUB_OFFLINE=1; export TOKENIZERS_PARALLELISM=false; cd $WS"
ROOT=$WS/artifacts/vla_pruner_openvla_reproduction/closed_loop
mkdir -p "$ROOT/logs"
ARMS=vanilla,vla_pruner_prune25,vla_pruner_prune50
RUN() { # gpu logname cmds...
  local gpu=$1 name=$2; shift 2
  local cmd="$ENV"
  for spec in "$@"; do
    set -- $spec
    cmd="$cmd; $PY research/semantic_token_cd/vla_pruner_closed_loop_rollout.py --task $1 --seeds $2 --arms $ARMS --gpu $gpu --worker-id $name --artifact-dir $ROOT --save-video"
  done
  cmd="$cmd > '$ROOT/logs/${name}.log' 2>&1"
  tmux kill-session -t "vp_${name}" 2>/dev/null || true
  tmux new-session -d -s "vp_${name}" "$cmd"
  echo "launched vp_${name} gpu=$gpu"
}
# W1 (g2): drawers then a few move scenes
RUN 2 w1 "google_robot_open_drawer 118" "google_robot_close_drawer 118" "google_robot_move_near 101,112,119"
# W2 (g2): pick chunk + a few move scenes
RUN 2 w2 "google_robot_pick_coke_can 100-102" "google_robot_move_near 120,123"
# W3 (g3): pick chunk + a few move scenes
RUN 3 w3 "google_robot_pick_coke_can 103-106" "google_robot_move_near 131,133"
# W4 (g3): remaining pick + move scenes
RUN 3 w4 "google_robot_pick_coke_can 107-109" "google_robot_move_near 135,136,152"
