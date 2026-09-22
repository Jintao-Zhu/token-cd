#!/usr/bin/env bash
set -uo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
CAL="$WS/artifacts/l11_entity_budget_calibration_v1"
LOG="$CAL/logs/auto_chain.log"
mkdir -p "$CAL/logs"; cd "$WS"; exec > >(tee -a "$LOG") 2>&1
status_json(){ "$PY" - "$CAL" "$1" "$2" <<'PY'
import json,os,sys
from datetime import datetime,timezone
from pathlib import Path
root,stage,msg=Path(sys.argv[1]),sys.argv[2],sys.argv[3]
p={"stage":stage,"message":msg,"updated_at":datetime.now(timezone.utc).astimezone().isoformat()}
t=root/f"QUEUE_STATUS.{os.getpid()}.tmp"; t.write_text(json.dumps(p,indent=2)+"\n"); os.replace(t,root/"QUEUE_STATUS.json")
PY
}
echo "[$(date '+%F %T')] entity calibration watcher started"
for _ in $(seq 1 720); do
  if ! pgrep -f '[o]rchestrate_extract_entity_budget_calibration.sh' >/dev/null 2>&1; then break; fi
  sleep 30
done
if pgrep -f '[o]rchestrate_extract_entity_budget_calibration.sh' >/dev/null 2>&1; then status_json failed "calibration orchestration still running after 6 hours"; exit 1; fi
N=$(find "$CAL/states" -name 'seed_*_step_*.json' | wc -l)
echo "[$(date '+%F %T')] calibration extraction exited; states=$N"
if [[ "$N" -ne 800 ]]; then status_json failed "expected 800 calibration states, got $N"; exit 1; fi
status_json analysis "calibration states complete; running tau calibration"
if ! "$PY" research/semantic_token_cd/analyze_entity_budget_calibration.py --artifact "$CAL"; then status_json failed "tau calibration failed"; exit 1; fi
[[ -f "$CAL/CALIBRATION_RESULTS.json" ]] || { status_json failed "missing CALIBRATION_RESULTS.json"; exit 1; }
status_json rollout "tau frozen; starting Entity-TopP 400-episode closed loop"
bash research/semantic_token_cd/run_l11_entity_budget_closed_loop.sh || { status_json failed "Entity-TopP closed loop failed"; exit 1; }
status_json complete "Entity-TopP stage complete; Generic stage intentionally not started"
echo "[$(date '+%F %T')] Entity-TopP stage complete"
