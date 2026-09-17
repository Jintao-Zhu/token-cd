#!/usr/bin/env bash
set -euo pipefail
cd /home/leju-suzhou/zjt_ws/token-cd
# The old coordinator is deliberately SIGSTOP'ed before this script starts.
# Its six already-launched workers finish their current five-seed chunks
# normally. Only then is the coordinator replaced, so no task/seed can be
# dispatched by both generations.
while pgrep -f '/xdiff_rollout.py --task' >/dev/null; do
  sleep 10
done
pkill -KILL -f '^/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python research/semantic_token_cd/xdiff_coordinator.py --workers-per-gpu 3$' 2>/dev/null || true
tmux kill-session -t xdiff_closed 2>/dev/null || true
tmux new-session -d -s xdiff_closed_expanded \
  "cd /home/leju-suzhou/zjt_ws/token-cd && PYTHONPATH=/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source /home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python research/semantic_token_cd/xdiff_coordinator.py --primary-gpus 2,3 --workers-per-gpu 3 --aux-gpus 1,4,5 --aux-workers-per-gpu 1 2>&1 | tee -a artifacts/prompt_attn_semantic_difference_v1/closed_loop/coordinator_expanded.log"
