#!/usr/bin/env bash
set -uo pipefail

WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
EXTRACT_ROOT="$WS/artifacts/l11_budget_estimators_v1"
CLOSED_ROOT="$WS/artifacts/l11_budget_estimators_closed_loop_100_199_v1"
LOG="$EXTRACT_ROOT/logs/auto_chain.log"
mkdir -p "$EXTRACT_ROOT/logs" "$CLOSED_ROOT/logs"
cd "$WS"
exec > >(tee -a "$LOG") 2>&1

status_json() {
  local root="$1" stage="$2" message="$3"
  "$PY" - "$root" "$stage" "$message" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
root, stage, message = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
payload = {"stage": stage, "message": message, "updated_at": datetime.now(timezone.utc).astimezone().isoformat()}
tmp = root / f"QUEUE_STATUS.{os.getpid()}.tmp"
tmp.write_text(json.dumps(payload, indent=2) + "\n")
os.replace(tmp, root / "QUEUE_STATUS.json")
PY
}

echo "[$(date '+%F %T')] watcher started; waiting for extract orchestrator to exit"
for _ in $(seq 1 480); do
  if ! pgrep -f '[o]rchestrate_extract_budget_estimators.sh' >/dev/null 2>&1; then
    break
  fi
  sleep 30
done

if pgrep -f '[o]rchestrate_extract_budget_estimators.sh' >/dev/null 2>&1; then
  status_json "$EXTRACT_ROOT" failed "extract orchestration still running after 4 hours"
  exit 1
fi

count=$(find "$EXTRACT_ROOT/estimators" -type f -name 'seed_*.json' | wc -l)
echo "[$(date '+%F %T')] extract orchestrator exited; seed files=$count (expected 200)"
if [[ "$count" -ne 200 ]]; then
  status_json "$EXTRACT_ROOT" failed "extract incomplete: expected 200 seed files, got $count"
  exit 1
fi

status_json "$EXTRACT_ROOT" analysis "extraction complete; running offline budget-estimator comparison"
if ! "$PY" research/semantic_token_cd/analyze_budget_estimators.py --artifact "$EXTRACT_ROOT"; then
  status_json "$EXTRACT_ROOT" failed "offline budget-estimator analysis failed"
  exit 1
fi
if [[ ! -f "$EXTRACT_ROOT/OFFLINE_RESULTS.json" ]]; then
  status_json "$EXTRACT_ROOT" failed "offline analysis produced no OFFLINE_RESULTS.json"
  exit 1
fi
status_json "$EXTRACT_ROOT" complete "offline analysis complete; starting closed-loop Relative/Spectral experiment"

echo "[$(date '+%F %T')] starting closed-loop Relative/Spectral experiment"
if bash research/semantic_token_cd/run_l11_budget_estimators_closed_loop.sh; then
  status_json "$CLOSED_ROOT" complete "closed-loop Relative/Spectral experiment complete"
  echo "[$(date '+%F %T')] closed-loop experiment complete"
else
  status_json "$CLOSED_ROOT" failed "closed-loop Relative/Spectral experiment failed"
  echo "[$(date '+%F %T')] closed-loop experiment failed"
  exit 1
fi
