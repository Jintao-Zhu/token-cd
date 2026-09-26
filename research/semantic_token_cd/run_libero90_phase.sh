#!/usr/bin/env bash
set -euo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
ROOT="$WS/artifacts/libero90_phase_gating_v1"
CONFIG="$ROOT/FROZEN_PHASE_CONFIG.json"
CKPT=/home/leju-suzhou/zjt_ws/checkpoints/libero/vq-vla-openvla-7b-libero90
REV=794ef81b7be928ea9270e81ca1ef5b60ffa9420f
MODE="${1:-status}"
count_cases(){ find "$1" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l; }
worker_pids(){ ps -eo pid,args | awk '$2 ~ /python$/ && $0 ~ /libero90_phase_worker.py/ {print $1}'; }
case "$MODE" in
 start|resume)
  if [ -n "$(worker_pids)" ]; then echo 'phase workers already running' >&2; exit 2; fi
  if [ "$MODE" = resume ]; then python3 - <<'PY'
import os
from pathlib import Path
r=Path('/home/leju-suzhou/zjt_ws/token-cd/artifacts/libero90_phase_gating_v1/cases')
p=r/'pending'; p.mkdir(parents=True,exist_ok=True)
for x in (r/'running').glob('*.json'):
 t=p/(x.name.split('__worker')[0]+'.json')
 if not t.exists(): os.replace(x,t)
PY
  fi
  for spec in '1 phase_g1a 1' '1 phase_g1b 1' '2 phase_g2a 1' '2 phase_g2b 1' '3 phase_g3a 1' '3 phase_g3b 1'; do
   set -- $spec; g=$1; wid=$2; render=$3; if [ "$g" -eq 1 ]; then vis=1; else vis="1,$g"; fi
   tmux kill-session -t "libero90_${wid}" 2>/dev/null || true
   tmux new-session -d -s "libero90_${wid}" "cd '$WS'; source scripts/activate_libero_openvla.sh; export CUDA_VISIBLE_DEVICES=${vis} MUJOCO_EGL_DEVICE_ID=${render}; export TOKENIZERS_PARALLELISM=false; python research/semantic_token_cd/libero90_phase_worker.py --artifact '$ROOT' --formal-root '$WS/artifacts/libero90_five_task_simpler_config_v1' --config '$CONFIG' --checkpoint '$CKPT' --checkpoint-revision '$REV' --physical-gpu ${g} --render-gpu ${render} --worker-id ${wid} --max-cases 0 --save-video > '$ROOT/logs/driver_${wid}.log' 2>&1; echo exit_code=\$? >> '$ROOT/logs/driver_${wid}.log'"
  done
  echo 'started 6 phase workers';;
 stop)
  for p in $(worker_pids); do kill "$p" 2>/dev/null || true; done
  for wid in phase_g1a phase_g1b phase_g2a phase_g2b phase_g3a phase_g3b; do tmux kill-session -t "libero90_${wid}" 2>/dev/null || true; done
  echo 'stopped phase workers';;
 status)
  echo "pending=$(count_cases "$ROOT/cases/pending") running=$(count_cases "$ROOT/cases/running") completed=$(count_cases "$ROOT/cases/completed") error=$(count_cases "$ROOT/cases/error") pairs=$(find "$ROOT/pairs" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)"
  echo "workers:"; worker_pids | tr '\n' ' '; echo
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk 'NR<=4';;
 *) echo 'usage: start|resume|stop|status' >&2; exit 2;;
esac
