#!/usr/bin/env bash
# DTP calibration v1 -- one worker per physical GPU (Vulkan safety).
set -u
WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ENV="export PYTHONPATH=$WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source; export HF_HUB_OFFLINE=1; export TOKENIZERS_PARALLELISM=false; cd $WS"
ROOT=$WS/artifacts/dtp_openvla_calibration_v1/closed_loop
mkdir -p "$ROOT/logs"
ARMS=control,l11_k64_t05,l7_k64_t05
W=(
 "google_robot_close_drawer 118 1 close_118"
 "google_robot_pick_coke_can 100-103 2 pick_a"
 "google_robot_pick_coke_can 104-109 3 pick_b"
 "google_robot_move_near 101,112,119,120,123 1 move_a"
 "google_robot_move_near 131,133,135,136,152 4 move_b"
)
for spec in "${W[@]}"; do
  set -- $spec
  task=$1; seeds=$2; gpu=$3; wid=$4
  name="dtp_cal_${wid}"
  tmux kill-session -t "$name" 2>/dev/null || true
  cmd="$ENV; $PY research/semantic_token_cd/dtp_closed_loop_rollout.py --task $task --seeds $seeds --arms $ARMS --gpu $gpu --worker-id $wid --artifact-dir $ROOT --save-video > '$ROOT/logs/${wid}.log' 2>&1"
  tmux new-session -d -s "$name" "$cmd"
  echo "launched $name gpu=$gpu task=$task seeds=$seeds"
done
