#!/usr/bin/env bash
# Closeout for the TopP90 replication: wait until 800 episodes exist or workers are gone,
# then write the verdict analysis.
set -uo pipefail
REPO=/home/leju-suzhou/zjt_ws/token-cd
ART="$REPO/artifacts/openvla_topp90_repl_100_199_v1"
LOG="$ART/logs/closeout.log"
mkdir -p "$ART/logs"
source "$REPO/scripts/activate_simpler_local.sh"
cd "$REPO"
echo "[$(date -Is)] topp90 closeout watcher started (pid $$)" >> "$LOG"
while true; do
  n=$(find "$ART/episodes" -name 'episode_*_summary.json' 2>/dev/null | wc -l)
  w=$(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -c '^ovrep_g' || true)
  echo "[$(date -Is)] episodes=$n/800 workers=$w" >> "$LOG"
  if [ "$n" -ge 800 ] || [ "$w" -eq 0 ]; then
    python scripts/analyze_openvla_topp90_repl.py --write --bootstrap 10000 >> "$LOG" 2>&1
    echo "$?" > "$ART/CLOSEOUT_EXIT"
    echo "[$(date -Is)] closeout exit=$(cat "$ART/CLOSEOUT_EXIT")" >> "$LOG"
    exit 0
  fi
  sleep 300
done
