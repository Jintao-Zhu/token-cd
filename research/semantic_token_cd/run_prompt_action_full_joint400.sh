#!/usr/bin/env bash
set -euo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd; PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python; ROOT="$WS/artifacts/prompt_action_full_joint_top2k_v1"
export PYTHONPATH="$WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/logs"; cd "$WS"
run(){ "$PY" research/semantic_token_cd/prompt_action_full_joint_rollout.py --task "$2" --seeds "$3" --gpu "$1" --worker-id "$4" --artifact "$ROOT"; }
case "${1:-}" in
 g2a) run 2 google_robot_open_drawer 0-66 g2a;; g2b) run 2 google_robot_close_drawer 0-66 g2b;; g2c) run 2 google_robot_pick_coke_can 0-65 g2c;;
 g3a) run 3 google_robot_open_drawer 67-99 g3a; run 3 google_robot_move_near 0-33 g3a;;
 g3b) run 3 google_robot_close_drawer 67-99 g3b; run 3 google_robot_move_near 34-67 g3b;;
 g3c) run 3 google_robot_pick_coke_can 66-99 g3c; run 3 google_robot_move_near 68-99 g3c;;
 finalize) while [ "$(find "$ROOT/closed_loop/episodes" -type f -name 'episode_*_summary.json' 2>/dev/null|wc -l)" -lt 400 ]; do sleep 30; done; "$PY" research/semantic_token_cd/analyze_prompt_action_full_joint.py > "$ROOT/logs/final_analysis.json";;
 *) echo "usage: $0 {g2a|g2b|g2c|g3a|g3b|g3c|finalize}" >&2; exit 2;; esac
