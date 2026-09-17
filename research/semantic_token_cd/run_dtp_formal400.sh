#!/usr/bin/env bash
# DTP calibration v1 -- formal fixed benchmark: frozen unified config l11_k64_t05
# over canonical seeds 0-99 x 4 tasks (400 episodes). Uses GPU2/3, 2 workers each.
set -u
WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ENV="export PYTHONPATH=$WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source; export HF_HUB_OFFLINE=1; export TOKENIZERS_PARALLELISM=false; cd $WS"
ROOT=$WS/artifacts/dtp_openvla_calibration_v1/formal_400
mkdir -p "$ROOT/logs"
ARM=l11_k64_t05
# task gpu
W=(
 "google_robot_open_drawer 2"
 "google_robot_close_drawer 3"
 "google_robot_pick_coke_can 2"
 "google_robot_move_near 3"
)
for spec in "${W[@]}"; do
  set -- $spec
  task=$1; gpu=$2
  name="dtp_formal_${task}"
  tmux kill-session -t "$name" 2>/dev/null || true
  cmd="$ENV; $PY research/semantic_token_cd/dtp_closed_loop_rollout.py --task $task --seeds 0-99 --arms $ARM --gpu $gpu --worker-id formal_$task --artifact-dir $ROOT > '$ROOT/logs/${task}.log' 2>&1"
  tmux new-session -d -s "$name" "$cmd"
  echo "launched $name gpu=$gpu task=$task"
done
