#!/usr/bin/env bash
set -euo pipefail

cd /home/leju-suzhou/zjt_ws/token-cd
artifact="artifacts/vla_pruner_openvla_reproduction/paper_v5_p75/formal"
log="$artifact/logs/finalizer.log"

while true; do
  count=$(find "$artifact/episodes" -type f -name '*_summary.json' 2>/dev/null | wc -l)
  printf '%s completed=%s/800\n' "$(date --iso-8601=seconds)" "$count" >> "$log"
  if [ "$count" -ge 800 ]; then
    break
  fi
  sleep 60
done

/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python \
  research/semantic_token_cd/vla_pruner_p75_analyze.py >> "$log" 2>&1
printf '%s analysis_complete\n' "$(date --iso-8601=seconds)" >> "$log"
