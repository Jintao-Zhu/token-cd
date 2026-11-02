#!/usr/bin/env bash
# Automatic closeout for the OpenVLA KMeans-granularity sweep.
#
# Waits for the queue supervisor to reach a terminal phase, then runs the analysis
# and figure generator with the full bootstrap.  Nothing here touches the running
# experiment; it only reads STATUS.json and, afterwards, episode summaries.
#
# Launch:  tmux new-session -d -s ovkm_closeout "bash scripts/closeout_kmeans_watch.sh"
set -uo pipefail

REPO=/home/leju-suzhou/zjt_ws/token-cd
ART="$REPO/artifacts/openvla_kmeans_budget_9task_200_299_v1"
SMOKE="$REPO/artifacts/openvla_kmeans_budget_smoke_v1"
LOG="$ART/logs/closeout.log"
mkdir -p "$ART/logs"

source "$REPO/scripts/activate_simpler_local.sh"
cd "$REPO"

echo "[$(date -Is)] closeout watcher started (pid $$)" >> "$LOG"
while true; do
  phase=$(python -c "import json;print(json.load(open('$ART/STATUS.json')).get('phase','unknown'))" 2>/dev/null || echo unreadable)
  case "$phase" in
    completed|failed)
      echo "[$(date -Is)] terminal phase=$phase; running analysis" >> "$LOG"
      if [ -f "$SMOKE/SMOKE_VALIDATION.json" ]; then
        cp "$SMOKE/SMOKE_VALIDATION.json" "$ART/SMOKE_VALIDATION.json"
      fi
      python scripts/analyze_openvla_kmeans_budget_study.py --write --bootstrap 10000 >> "$LOG" 2>&1
      status=$?
      echo "$status" > "$ART/CLOSEOUT_EXIT"
      echo "[$(date -Is)] analysis exit=$status" >> "$LOG"
      python - <<PY >> "$LOG" 2>&1
import json, os
from pathlib import Path
status = json.loads(Path("$ART/STATUS.json").read_text())
exit_file = Path("$ART/CLOSEOUT_EXIT")
final = {
    "closed_out_at": "$(date -Is)",
    "supervisor_phase": status.get("phase"),
    "completed_chunks": status.get("completed_chunks"),
    "failed_workers": status.get("failed_workers"),
    "analysis_exit_code": int(exit_file.read_text().strip()) if exit_file.is_file() else None,
    "smoke_validation_copied": os.path.exists("$ART/SMOKE_VALIDATION.json"),
}
Path("$ART/CLOSEOUT.json").write_text(json.dumps(final, indent=2) + "\n")
print(json.dumps(final))
PY
      exit 0
      ;;
    *)
      sleep 60
      ;;
  esac
done
