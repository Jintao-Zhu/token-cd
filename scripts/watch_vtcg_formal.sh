#!/usr/bin/env bash
# Unattended watchdog for the formal queue.
#
# It ONLY restarts the same queue command with --repair-running-cases when no queue
# process is alive and the phase is not yet complete. It never edits artifacts, never
# deletes anything, and never changes protocol parameters. Every restart is logged so
# the final report can state how many restarts happened.
set -uo pipefail
ROOT=/home/leju-suzhou/zjt_ws/token-cd
ART="$ROOT/artifacts/vtcg_factorial_controls_v1"
LOG="$ART/formal_watchdog.log"
QUEUE_ARGS=(--phase formal --workers-per-gpu 3 --repair-running-cases)

log() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >>"$LOG"; }

log "watchdog start pid=$$"
consecutive_failures=0
while true; do
  if pgrep -f "vtcg_factorial_queue.py" >/dev/null 2>&1; then
    sleep 60
    continue
  fi
  status=$(python3 -c "
import json,sys
try:
    print(json.load(open('$ART/QUEUE_STATUS.json')).get('status'))
except Exception as exc:
    print('unknown')
")
  if [ "$status" = "complete" ]; then
    log "queue complete; watchdog exiting"
    exit 0
  fi
  # A deterministic failure (e.g. a shared-source hash change) makes every restart fail
  # instantly; stop after 3 consecutive fast failures instead of looping forever, and
  # leave a marker so a human/agent investigates.
  now=$(date +%s)
  if [ -n "${last_restart:-}" ] && [ $((now - last_restart)) -lt 180 ]; then
    consecutive_failures=$((consecutive_failures + 1))
  else
    consecutive_failures=0
  fi
  if [ "$consecutive_failures" -ge 3 ]; then
    log "3 consecutive fast failures; STOPPING watchdog for manual audit"
    printf '%s\n' "{\"stopped_at\": $now, \"status\": \"$status\", \"reason\": \"repeated fast queue failures; manual audit required\"}" >"$ART/WATCHDOG_STOPPED.json"
    exit 1
  fi
  log "queue not running (status=$status); restarting with repair (attempt $((consecutive_failures + 1)))"
  "$ROOT/scripts/run_vtcg_factorial_queue.sh" "${QUEUE_ARGS[@]}" >>"$ART/formal_driver.log" 2>&1 &
  last_restart=$(date +%s)
  log "restarted pid=$!"
  sleep 60
done
