#!/usr/bin/env bash
set -euo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
ROOT="$WS/artifacts/dtp_fixed_positive_l11_matched_cd_v1"
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
export PYTHONPATH="$WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false DTP_L11_PREFLIGHT_GPU=2
mkdir -p "$ROOT/logs" "$ROOT/preflight"
while pgrep -f "prompt_action_full_joint_rollout.py.*--worker-id g2c" >/dev/null; do sleep 20; done
cd "$WS"
if "$PY" research/semantic_token_cd/preflight_dtp_l11_cd.py > "$ROOT/logs/preflight.log" 2>&1; then
  touch "$ROOT/preflight/PASS"
else
  touch "$ROOT/preflight/FAILED"
  exit 1
fi

