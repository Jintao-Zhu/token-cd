#!/usr/bin/env bash
set -euo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
ROOT="$WS/artifacts/libero90_five_task_simpler_config_v1"
CONFIG="$ROOT/FROZEN_METHOD_CONFIG.json"
CKPT=/home/leju-suzhou/zjt_ws/checkpoints/libero/vq-vla-openvla-7b-libero90
REV=794ef81b7be928ea9270e81ca1ef5b60ffa9420f
MODE="${1:-status}"

count_cases() {
  local d="$1"
  find "$d" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l
}

worker_pids() {
  ps -eo pid,args | awk '$2 ~ /python$/ && $0 ~ /libero90_formal_worker.py/ {print $1}'
}

case "$MODE" in
  start|resume)
    if [ -n "$(worker_pids)" ]; then
      echo "refusing to start: formal worker processes are already running" >&2
      exit 2
    fi
    if [ "$MODE" = resume ]; then
      python3 - <<'PY'
import os
from pathlib import Path
root=Path('/home/leju-suzhou/zjt_ws/token-cd/artifacts/libero90_five_task_simpler_config_v1')
pending=root/'cases'/'pending'; pending.mkdir(parents=True,exist_ok=True)
for p in (root/'cases'/'running').glob('*.json'):
    target=pending/(p.name.split('__worker')[0]+'.json')
    if not target.exists(): os.replace(p,target)
PY
    fi
    for spec in '1 g1a 1' '1 g1b 1' '2 g2a 1' '2 g2b 1' '3 g3a 1' '3 g3b 1'; do
      set -- $spec; g=$1; wid=$2; render=$3
      if [ "$g" -eq 1 ]; then vis=1; else vis="1,$g"; fi
      tmux kill-session -t "libero90_${wid}" 2>/dev/null || true
      tmux new-session -d -s "libero90_${wid}" "cd '$WS'; source scripts/activate_libero_openvla.sh; export CUDA_VISIBLE_DEVICES=${vis} MUJOCO_EGL_DEVICE_ID=${render}; export TOKENIZERS_PARALLELISM=false; python research/semantic_token_cd/libero90_formal_worker.py --artifact '$ROOT' --config '$CONFIG' --checkpoint '$CKPT' --checkpoint-revision '$REV' --physical-gpu ${g} --render-gpu ${render} --worker-id ${wid} --max-cases 0 --save-video > '$ROOT/logs/driver_${wid}.log' 2>&1; echo exit_code=\$? >> '$ROOT/logs/driver_${wid}.log'"
    done
    echo "started 6 formal workers"
    ;;
  stop)
    for p in $(worker_pids); do kill "$p" 2>/dev/null || true; done
    for wid in g1a g1b g2a g2b g3a g3b; do tmux kill-session -t "libero90_${wid}" 2>/dev/null || true; done
    echo "stopped formal workers"
    ;;
  status)
    echo "pending=$(count_cases "$ROOT/cases/pending") running=$(count_cases "$ROOT/cases/running") completed=$(count_cases "$ROOT/cases/completed") error=$(count_cases "$ROOT/cases/error") pairs=$(find "$ROOT/pairs" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)"
    echo "workers:"; worker_pids | tr '\n' ' '; echo
    echo "gpu:"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | head -4
    ;;
  *)
    echo "usage: $0 {start|resume|stop|status}" >&2; exit 2 ;;
esac
