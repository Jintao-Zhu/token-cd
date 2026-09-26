#!/usr/bin/env bash
set -u
TASK="$1"
GPU="$2"
WORKER_ID="$3"
ROOT=/home/leju-suzhou/zjt_ws/token-cd
ART="$ROOT/artifacts/task_conditioned_contrast_midsize_v1"
DB="$ART/jobs.sqlite"
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
PP=/home/leju-suzhou/zjt_ws/token-cd/task1/shim_site:/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
LOG="$ART/logs/formal_${WORKER_ID}.log"
mkdir -p "$ART/logs" "$ART/traces" "$ART/episodes"

count_pending() {
  /usr/bin/python3 - "$DB" "$TASK" <<'PY'
import sqlite3, sys
con = sqlite3.connect(sys.argv[1])
print(con.execute("select count(*) from jobs where task=? and status='pending'", (sys.argv[2],)).fetchone()[0])
con.close()
PY
}

cd "$ROOT"
echo "[wrapper] start task=$TASK gpu=$GPU worker=$WORKER_ID $(date '+%F %T')" >> "$LOG"
for attempt in 1 2 3; do
  env HF_HUB_OFFLINE=1 TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH="$PP" CUDA_VISIBLE_DEVICES="$GPU" \
    "$PY" research/semantic_token_cd/task_conditioned_contrast_worker.py \
      --gpu "$GPU" --worker-id "$WORKER_ID" --task "$TASK" --jobs "$DB" >> "$LOG" 2>&1
  rc=$?
  echo "[wrapper] worker exit rc=$rc attempt=$attempt $(date '+%F %T')" >> "$LOG"
  "$PY" research/semantic_token_cd/requeue_task_conditioned_jobs.py --db "$DB" --task "$TASK" --stale-seconds 0 >> "$LOG" 2>&1
  pending=$(count_pending)
  echo "[wrapper] pending=$pending after attempt=$attempt" >> "$LOG"
  if [ "$pending" -eq 0 ]; then
    echo "[wrapper] ALL_DONE task=$TASK $(date '+%F %T')" >> "$LOG"
    exit 0
  fi
  sleep 15
done
echo "[wrapper] STOPPED_WITH_PENDING task=$TASK pending=$(count_pending) $(date '+%F %T')" >> "$LOG"
exit 1
