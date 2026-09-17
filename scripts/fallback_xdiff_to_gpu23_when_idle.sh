#!/usr/bin/env bash
set -euo pipefail
cd /home/leju-suzhou/zjt_ws/token-cd
while pgrep -f '/xdiff_rollout.py --task' >/dev/null; do
  sleep 10
done
tmux kill-session -t xdiff_closed_expanded 2>/dev/null || true
tmux new-session -d -s xdiff_closed_gpu23 \
  "cd /home/leju-suzhou/zjt_ws/token-cd && PYTHONPATH=/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source /home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python research/semantic_token_cd/xdiff_coordinator.py --primary-gpus 2,3 --workers-per-gpu 3 2>&1 | tee -a artifacts/prompt_attn_semantic_difference_v1/closed_loop/coordinator_gpu23_fallback.log"
