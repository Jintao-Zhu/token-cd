#!/usr/bin/env bash
set -euo pipefail
repo=/home/leju-suzhou/zjt_ws/token-cd
artifact="$repo/artifacts/target_boost_token_causal_audit_v1"
mkdir -p "$artifact/logs"
export PYTHONPATH="$repo:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source:${PYTHONPATH:-}"
python_bin=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
"$python_bin" "$repo/research/semantic_token_cd/target_boost_token_causal_coordinator.py" \
  >"$artifact/logs/coordinator.log" 2>&1
"$python_bin" "$repo/research/semantic_token_cd/analyze_target_boost_token_causal.py" \
  >"$artifact/logs/analysis.log" 2>&1
