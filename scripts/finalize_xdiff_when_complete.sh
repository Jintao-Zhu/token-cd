#!/usr/bin/env bash
set -euo pipefail
cd /home/leju-suzhou/zjt_ws/token-cd
while [[ ! -f artifacts/prompt_attn_semantic_difference_v1/closed_loop/COMPLETE ]]; do
  sleep 30
done
export PYTHONPATH=/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python research/semantic_token_cd/analyze_xdiff_closed.py \
  > artifacts/prompt_attn_semantic_difference_v1/finalize.log 2>&1
