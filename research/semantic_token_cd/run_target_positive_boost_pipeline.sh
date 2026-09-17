#!/usr/bin/env bash
set -euo pipefail
repo=/home/leju-suzhou/zjt_ws/token-cd
artifact="$repo/artifacts/target_positive_boost_shr_v1"
export PYTHONPATH="$repo:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source:${PYTHONPATH:-}"
mkdir -p "$artifact/logs"
/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python \
  "$repo/research/semantic_token_cd/prepare_target_positive_boost_offline.py" \
  >"$artifact/logs/offline.log" 2>&1
if [[ ! -f "$artifact/OFFLINE_PASS" ]]; then exit 2; fi
/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python \
  "$repo/research/semantic_token_cd/target_specific_coordinator.py" \
  --variant positive_boost --gpus 2,3 --workers-per-gpu 3 \
  >"$artifact/logs/coordinator.log" 2>&1
/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python \
  "$repo/research/semantic_token_cd/analyze_target_specific_closed.py" \
  --variant positive_boost >"$artifact/logs/final_analysis.log" 2>&1
