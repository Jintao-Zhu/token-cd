#!/usr/bin/env bash
# Automatic closeout for the real K=10 run: wait until every planned episode exists (or the
# workers are gone), then write the paired analysis and the figure.
# Launch: tmux new-session -d -s ovk10_closeout "bash scripts/closeout_k10_watch.sh"
set -uo pipefail
REPO=/home/leju-suzhou/zjt_ws/token-cd
ART="$REPO/artifacts/openvla_k10_2lambda_200_299_v1"
TARGET=800
LOG="$ART/logs/closeout.log"
mkdir -p "$ART/logs"
source "$REPO/scripts/activate_simpler_local.sh"
cd "$REPO"

echo "[$(date -Is)] closeout watcher started (pid $$), target $TARGET episodes" >> "$LOG"
while true; do
  done_count=$(find "$ART/episodes" -name 'episode_*_summary.json' 2>/dev/null | wc -l)
  workers=$(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -c '^ovk10_' || true)
  echo "[$(date -Is)] episodes=$done_count/$TARGET workers=$workers" >> "$LOG"
  if [ "$done_count" -ge "$TARGET" ] || [ "$workers" -eq 0 ]; then
    echo "[$(date -Is)] run finished (episodes=$done_count, workers=$workers); analysing" >> "$LOG"
    python scripts/analyze_openvla_k10_study.py --write --bootstrap 10000 >> "$LOG" 2>&1
    status=$?
    echo "$status" > "$ART/CLOSEOUT_EXIT"
    python - <<PY >> "$LOG" 2>&1
import json, glob
from pathlib import Path
final = {
    "closed_out_at": "$(date -Is)",
    "episodes_found": len(glob.glob("$ART/episodes/*/*/episode_*_summary.json")),
    "analysis_exit_code": int(Path("$ART/CLOSEOUT_EXIT").read_text().strip()),
}
Path("$ART/CLOSEOUT.json").write_text(json.dumps(final, indent=2) + "\n")
print(json.dumps(final))
PY
    echo "[$(date -Is)] closeout done exit=$status" >> "$LOG"
    exit 0
  fi
  sleep 300
done
