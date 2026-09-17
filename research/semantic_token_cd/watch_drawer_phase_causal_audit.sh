#!/usr/bin/env bash
set -euo pipefail
artifact="$1"
expected="${2:-48}"
mkdir -p "$artifact/results"
while true; do
  count=$(find "$artifact/results" -type f -name '*.json' 2>/dev/null | wc -l)
  if [[ "$count" -ge "$expected" ]]; then
    break
  fi
  sleep 30
done
/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python \
  /home/leju-suzhou/zjt_ws/token-cd/research/semantic_token_cd/analyze_drawer_phase_causal_audit.py \
  --artifact "$artifact"
