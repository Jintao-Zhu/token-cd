#!/usr/bin/env bash
set -euo pipefail
root=/home/leju-suzhou/zjt_ws/token-cd/artifacts/l11_token_group_causal_audit_v1
while true; do
  expected=$(/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python -c "import json; print(len(json.load(open('$root/MANIFEST.json'))['cases']))")
  actual=$(find "$root/results" -name '*.json' 2>/dev/null | wc -l)
  if [ "$actual" -eq "$expected" ]; then
    cd /home/leju-suzhou/zjt_ws/token-cd
    export PYTHONPATH=/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
    exec /home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python research/semantic_token_cd/analyze_l11_token_causal_audit.py --artifact "$root"
  fi
  sleep 30
done
