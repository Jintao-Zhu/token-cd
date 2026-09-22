#!/usr/bin/env bash
set -uo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ROOT="$WS/artifacts/matched_dynamic_budget_analysis_v1"
CAL="$WS/artifacts/l11_entity_budget_calibration_v1"
MATCHED="$WS/artifacts/prompt_attn_l11_token_count_v1"
SHUFFLE="$WS/artifacts/l11_matched_state_coupling_v1"
LOG="$ROOT/logs/auto_analysis.log"; mkdir -p "$ROOT/logs"; cd "$WS"; exec > >(tee -a "$LOG") 2>&1
echo "[$(date '+%F %T')] matched-support analysis watcher started"
for _ in $(seq 1 480); do
  if ! pgrep -f '[o]rchestrate_extract_matched_support_metrics.sh' >/dev/null 2>&1; then break; fi
  sleep 30
done
N=$(find "$ROOT/metrics" -name 'seed_*_step_*.json' 2>/dev/null | wc -l)
echo "[$(date '+%F %T')] metrics extraction exited; files=$N"
if [[ "$N" -ne 800 ]]; then echo "incomplete metrics; analysis not run" >&2; exit 1; fi
"$PY" research/semantic_token_cd/analyze_matched_dynamic_budget.py \
  --artifact "$ROOT" --calibration "$CAL/states" \
  --calibration-results "$CAL/CALIBRATION_RESULTS.json" \
  --matched "$MATCHED" --shuffle "$SHUFFLE"
echo "[$(date '+%F %T')] mechanism analysis complete"
