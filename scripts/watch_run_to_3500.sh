#!/usr/bin/env bash
# Passive watchdog for the pi0-FAST L6-matched run: logs state every 2 minutes until the
# run reaches 3500 episodes or the workers vanish, then writes a terminal marker and exits.
# It NEVER starts, kills or restarts anything - the session goal does the reacting.
R=/home/leju-suzhou/zjt_ws/token-cd/runs/pi0fast_l6_matched_9task_100seed
LOG=$R/logs/watchdog.log
mkdir -p "$R/logs"
exec >> "$LOG" 2>&1
echo "=== watchdog started $(date '+%F %T') (target 3500) ==="
for i in $(seq 1 240); do                     # up to 8 h
  n=$(find "$R/episodes" -name '*.json' 2>/dev/null | wc -l)
  w=$(pgrep -cf "[p]i0_fast_simpler_l6_matched_rollout" || true)
  s=$(pgrep -cf "[p]i0_fast_l6_simpler_server" || true)
  echo "$(date '+%F %T') episodes=$n workers=$w servers=$s"
  if [ "$n" -ge 3500 ]; then
    echo "$(date '+%F %T') COMPLETE episodes=$n"
    break
  fi
  if [ "$w" -eq 0 ] && [ "$n" -gt 100 ]; then
    echo "$(date '+%F %T') STALLED episodes=$n workers=0 - needs attention"
    break
  fi
  sleep 120
done
echo "=== watchdog exited $(date '+%F %T') ==="
