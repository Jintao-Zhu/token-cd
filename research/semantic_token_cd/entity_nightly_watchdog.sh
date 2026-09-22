#!/usr/bin/env bash
set -uo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
CAL="$WS/artifacts/l11_entity_budget_calibration_v1"
CLOSED="$WS/artifacts/l11_entity_budget_closed_loop_100_199_v1"
log="$CAL/logs/watchdog.log"; mkdir -p "$CAL/logs"; cd "$WS"
echo "[$(date '+%F %T')] entity watchdog started" >> "$log"
for _ in $(seq 1 1440); do
  # Stop only after the Entity stage has written a complete status.
  if [[ -f "$CLOSED/QUEUE_STATUS.json" ]] && grep -q '"stage": "complete"' "$CLOSED/QUEUE_STATUS.json"; then
    echo "[$(date '+%F %T')] closed-loop complete; watchdog exiting" >> "$log"; exit 0
  fi
  if pgrep -f '[w]atch_entity_calibration_then_run_entity.sh' >/dev/null 2>&1; then
    sleep 60; continue
  fi
  if pgrep -f '[e]xtract_entity_budget_calibration.py' >/dev/null 2>&1; then
    sleep 60; continue
  fi
  if pgrep -f '[p]rompt_attn_l11_entity_budget_rollout.py' >/dev/null 2>&1; then
    sleep 60; continue
  fi
  # Resume from the top-level watcher. It skips completed state files and
  # completed closed-loop episodes, so a restart is idempotent.
  echo "[$(date '+%F %T')] top-level watcher absent; restarting it" >> "$log"
  tmux new-session -d -s entity_auto "bash $WS/research/semantic_token_cd/watch_entity_calibration_then_run_entity.sh"
  sleep 60
done
echo "[$(date '+%F %T')] watchdog timeout" >> "$log"
