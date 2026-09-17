#!/usr/bin/env bash
# Prepared launcher only. Do not invoke until the user starts the formal run.
set -euo pipefail

PA_WS=/home/leju-suzhou/zjt_ws/token-cd
PA_PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
PA_ARTIFACT="$PA_WS/artifacts/prompt_action_rerank_l11_matched_v1"
export PYTHONPATH="$PA_WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$PA_ARTIFACT/logs"
cd "$PA_WS"

run_two() {
  local worker=$1 gpu=$2 task1=$3 seeds1=$4 task2=$5 seeds2=$6
  "$PA_PY" research/semantic_token_cd/prompt_action_rerank_rollout.py \
    --task "$task1" --seeds "$seeds1" --gpu "$gpu" --worker-id "$worker" \
    --artifact "$PA_ARTIFACT"
  "$PA_PY" research/semantic_token_cd/prompt_action_rerank_rollout.py \
    --task "$task2" --seeds "$seeds2" --gpu "$gpu" --worker-id "$worker" \
    --artifact "$PA_ARTIFACT"
}

run_one() {
  local worker=$1 gpu=$2 task=$3 seeds=$4
  "$PA_PY" research/semantic_token_cd/prompt_action_rerank_rollout.py \
    --task "$task" --seeds "$seeds" --gpu "$gpu" --worker-id "$worker" \
    --artifact "$PA_ARTIFACT"
}

case "${1:-}" in
  g2a) run_one g2a 2 google_robot_open_drawer 0-66 ;;
  g2b) run_one g2b 2 google_robot_close_drawer 0-66 ;;
  g2c) run_one g2c 2 google_robot_pick_coke_can 0-65 ;;
  g3a) run_two g3a 3 google_robot_open_drawer 67-99 google_robot_move_near 0-33 ;;
  g3b) run_two g3b 3 google_robot_close_drawer 67-99 google_robot_move_near 34-67 ;;
  g3c) run_two g3c 3 google_robot_pick_coke_can 66-99 google_robot_move_near 68-99 ;;
  finalize)
    while [ "$(find "$PA_ARTIFACT/closed_loop/episodes" -type f -path '*/pa_rerank_l11_matched/episode_*_summary.json' 2>/dev/null | wc -l)" -lt 400 ]; do
      sleep 30
    done
    "$PA_PY" research/semantic_token_cd/analyze_prompt_action_rerank.py \
      > "$PA_ARTIFACT/logs/final_analysis.json"
    ;;
  *) echo "usage: $0 {g2a|g2b|g2c|g3a|g3b|g3c|finalize}" >&2; exit 2 ;;
esac
