#!/usr/bin/env bash
set -euo pipefail
PA_WS=/home/leju-suzhou/zjt_ws/token-cd
PA_PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
PA_ROOT="$PA_WS/artifacts/prompt_action_constrained_joint_l11_v1"
export PYTHONPATH="$PA_WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$PA_ROOT/logs"
cd "$PA_WS"

run() {
  "$PA_PY" research/semantic_token_cd/prompt_action_joint_rollout.py \
    --task "$2" --seeds "$3" --arm "$4" --gpu "$1" --worker-id "$5" --artifact "$PA_ROOT"
}
case "${1:-}" in
  g2a) run 2 google_robot_open_drawer 0-99 pa_a10 g2a; run 2 google_robot_open_drawer 0-33 pa_joint10 g2a ;;
  g2b) run 2 google_robot_close_drawer 0-99 pa_a10 g2b; run 2 google_robot_open_drawer 34-66 pa_joint10 g2b ;;
  g2c) run 2 google_robot_pick_coke_can 0-99 pa_a10 g2c; run 2 google_robot_open_drawer 67-99 pa_joint10 g2c ;;
  g3a) run 3 google_robot_move_near 0-99 pa_a10 g3a; run 3 google_robot_close_drawer 0-33 pa_joint10 g3a ;;
  g3b) run 3 google_robot_pick_coke_can 0-99 pa_joint10 g3b; run 3 google_robot_close_drawer 34-66 pa_joint10 g3b ;;
  g3c) run 3 google_robot_move_near 0-99 pa_joint10 g3c; run 3 google_robot_close_drawer 67-99 pa_joint10 g3c ;;
  finalize)
    while [ "$(find "$PA_ROOT/closed_loop/episodes" -type f -name 'episode_*_summary.json' 2>/dev/null | wc -l)" -lt 800 ]; do sleep 30; done
    "$PA_PY" research/semantic_token_cd/analyze_prompt_action_joint.py > "$PA_ROOT/logs/final_analysis.json"
    ;;
  *) echo "usage: $0 {g2a|g2b|g2c|g3a|g3b|g3c|finalize}" >&2; exit 2 ;;
esac
