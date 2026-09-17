#!/usr/bin/env bash
set -euo pipefail
repo=/home/leju-suzhou/zjt_ws/token-cd
artifact="$repo/artifacts/target_specific_attention_shr_v1"
export PYTHONPATH="$repo:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source:${PYTHONPATH:-}"
mkdir -p "$artifact/logs"
while pgrep -f '[l]11_high_lambda_rollout.py' >/dev/null; do sleep 30; done
(
  env PYTHONPATH="$repo:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source" \
    /home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python "$repo/research/semantic_token_cd/target_specific_offline.py" \
    --task google_robot_open_drawer --gpu 2 >"$artifact/logs/offline_open.log" 2>&1
  env PYTHONPATH="$repo:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source" \
    /home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python "$repo/research/semantic_token_cd/target_specific_offline.py" \
    --task google_robot_pick_coke_can --gpu 2 >"$artifact/logs/offline_pick.log" 2>&1
) &
left=$!
(
  env PYTHONPATH="$repo:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source" \
    /home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python "$repo/research/semantic_token_cd/target_specific_offline.py" \
    --task google_robot_close_drawer --gpu 3 >"$artifact/logs/offline_close.log" 2>&1
  env PYTHONPATH="$repo:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source" \
    /home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python "$repo/research/semantic_token_cd/target_specific_offline.py" \
    --task google_robot_move_near --gpu 3 >"$artifact/logs/offline_move.log" 2>&1
) &
right=$!
wait "$left" "$right"
/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python "$repo/research/semantic_token_cd/analyze_target_specific_offline.py" \
  >"$artifact/logs/offline_analysis.log" 2>&1
if [[ ! -f "$artifact/OFFLINE_PASS" ]]; then exit 2; fi
env PYTHONPATH="$repo:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source" \
  /home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python "$repo/research/semantic_token_cd/target_specific_coordinator.py" \
  --gpus 2,3 --workers-per-gpu 3 >"$artifact/logs/coordinator.log" 2>&1
/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python "$repo/research/semantic_token_cd/analyze_target_specific_closed.py" \
  >"$artifact/logs/final_analysis.log" 2>&1
