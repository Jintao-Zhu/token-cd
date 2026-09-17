#!/usr/bin/env bash
set -euo pipefail
repo=/home/leju-suzhou/zjt_ws/token-cd
artifact="$repo/artifacts/target_positive_boost_confidence_gate_v1"
mkdir -p "$artifact/logs"
export PYTHONPATH="$repo:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source:${PYTHONPATH:-}"
py=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
CUDA_VISIBLE_DEVICES=2 "$py" "$repo/research/semantic_token_cd/preflight_target_boost_confidence.py" >"$artifact/logs/preflight.log" 2>&1
test -f "$artifact/PREFLIGHT_PASS"
"$py" "$repo/research/semantic_token_cd/target_boost_confidence_coordinator.py" >"$artifact/logs/coordinator.log" 2>&1
"$py" "$repo/research/semantic_token_cd/analyze_target_boost_confidence.py" >"$artifact/logs/analysis.log" 2>&1
