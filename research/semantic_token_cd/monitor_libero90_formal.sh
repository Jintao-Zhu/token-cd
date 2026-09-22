#!/usr/bin/env bash
set +e
ROOT=/home/leju-suzhou/zjt_ws/token-cd/artifacts/libero90_five_task_simpler_config_v1
while true; do
  pids=$(ps -eo pid,args | awk '$2 ~ /python$/ && $0 ~ /libero90_formal_worker.py/ {print $1}' | tr '\n' ' ')
  pending=$(find "$ROOT/cases/pending" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)
  running=$(find "$ROOT/cases/running" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)
  completed=$(find "$ROOT/cases/completed" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)
  errors=$(find "$ROOT/cases/error" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)
  echo "=== $(date -Iseconds) pending=$pending running=$running completed=$completed error=$errors workers_pids=$pids"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk 'NR<=4 {print}'
  if [ -z "$pids" ]; then
    echo "=== monitor_done $(date -Iseconds)"
    break
  fi
  sleep 60
done
