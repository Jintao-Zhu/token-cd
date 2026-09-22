#!/usr/bin/env bash
set -uo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ROOT="$WS/artifacts/intervention_dose_response_v1"
LOG="$ROOT/logs/auto_analysis.log"; mkdir -p "$ROOT/logs"; cd "$WS"; exec > >(tee -a "$LOG") 2>&1
echo "[$(date '+%F %T')] dose-response analysis watcher started"
for _ in $(seq 1 720); do
  if ! pgrep -f '[o]rchestrate_extract_intervention_dose_response.sh' >/dev/null 2>&1; then break; fi
  sleep 30
done
N=$(find "$ROOT/dose" -name 'seed_*.json' 2>/dev/null | wc -l)
echo "[$(date '+%F %T')] dose extraction exited; seed files=$N"
if [[ "$N" -ne 200 ]]; then echo "incomplete dose scan; analysis not run" >&2; exit 1; fi
"$PY" research/semantic_token_cd/analyze_intervention_dose_response.py --artifact "$ROOT"
echo "[$(date '+%F %T')] dose-response analysis complete"
