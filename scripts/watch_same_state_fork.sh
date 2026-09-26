#!/usr/bin/env bash
# Watchdog: keep fork workers alive, requeue transient failures, analyze when drained.
set -u
WS=/home/leju-suzhou/zjt_ws/token-cd
ROOT="$WS/artifacts/same_state_fork_v1"
LOG="$ROOT/logs/watchdog.log"
PY="/home/leju-suzhou/zjt_ws/libero-openvla-env/bin/python"
cd "$WS"
export PYTHONPATH="$WS"

count(){ find "$1" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l; }
alive(){ ps -eo args | grep -c '[l]ibero_same_state_fork_worker.py'; }

echo "[watch] start $(date '+%F %T')" >> "$LOG"
while true; do
  pend=$(count "$ROOT/cases/pending"); run=$(count "$ROOT/cases/running")
  err=$(count "$ROOT/cases/error");      done_=$(count "$ROOT/cases/completed")
  n=$(alive)
  echo "[watch] $(date '+%T') pending=$pend running=$run completed=$done_ error=$err workers=$n" >> "$LOG"

  if [ "$err" -gt 0 ]; then "$PY" scripts/requeue_forks.py >> "$LOG" 2>&1; fi

  # periodic partial analysis so intermediate conclusions are always available
  if [ "$done_" -gt 0 ] && [ $(( $(date +%s) % 1800 )) -lt 200 ]; then
    "$PY" research/semantic_token_cd/analyze_same_state_fork.py \
      --artifact "$ROOT" --clean-root "$WS/artifacts/libero90_five_task_simpler_config_v1" \
      --out "$ROOT/ANALYSIS_PARTIAL.json" >> "$LOG" 2>&1
  fi

  if [ "$n" -eq 0 ]; then
    if [ "$pend" -eq 0 ] && [ "$run" -eq 0 ]; then
      echo "[watch] drained $(date '+%T'); analyzing" >> "$LOG"
      "$PY" research/semantic_token_cd/analyze_same_state_fork.py \
        --artifact "$ROOT" --clean-root "$WS/artifacts/libero90_five_task_simpler_config_v1" >> "$LOG" 2>&1
      echo "[watch] ALL_DONE $(date '+%F %T')" >> "$LOG"
      break
    fi
    echo "[watch] restarting workers $(date '+%T')" >> "$LOG"
    bash research/semantic_token_cd/run_same_state_fork.sh start >> "$LOG" 2>&1
  fi
  sleep 180
done
