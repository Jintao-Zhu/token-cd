#!/usr/bin/env bash
# Launcher for SAME_STATE_FORK_V1 (LIBERO side).
set -euo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
ROOT="$WS/artifacts/same_state_fork_v1"
CKPT=/home/leju-suzhou/zjt_ws/checkpoints/libero/vq-vla-openvla-7b-libero90
REV=794ef81b7be928ea9270e81ca1ef5b60ffa9420f
MODE="${1:-status}"

count(){ find "$1" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l; }
pids(){ ps -eo pid,args | awk '$2 ~ /python$/ && $0 ~ /libero_same_state_fork_worker.py/ {print $1}'; }

case "$MODE" in
 start)
  # All workers render on GPU 1: the phase experiment showed that concurrent
  # offscreen rendering on several devices triggers
  # "Offscreen framebuffer is not complete".  Compute stays on each own GPU.
  for spec in "1 fork_g1a" "1 fork_g1b" "1 fork_g1c" "1 fork_g1d" "2 fork_g2a" "2 fork_g2b" "2 fork_g2c" "2 fork_g2d" "3 fork_g3a" "3 fork_g3b" "3 fork_g3c" "3 fork_g3d"; do
   set -- $spec; g=$1; wid=$2
   tmux kill-session -t "fork_${wid}" 2>/dev/null || true
   if [ "$g" -eq 1 ]; then vis=1; else vis="1,$g"; fi
   tmux new-session -d -s "fork_${wid}" \
    "cd '$WS'; source scripts/activate_libero_openvla.sh; export CUDA_VISIBLE_DEVICES=${vis} MUJOCO_EGL_DEVICE_ID=1 TOKENIZERS_PARALLELISM=false; \
     python research/semantic_token_cd/libero_same_state_fork_worker.py \
       --artifact '$ROOT' --manifest '$ROOT/LIBERO_STATE_MANIFEST.json' --config '$ROOT/FROZEN_PROTOCOL.json' \
       --checkpoint '$CKPT' --checkpoint-revision '$REV' --physical-gpu ${g} --render-gpu 1 --worker-id ${wid} \
       --max-cases 0 > '$ROOT/logs/driver_${wid}.log' 2>&1; echo exit=\$? >> '$ROOT/logs/driver_${wid}.log'"
  done
  echo "started 12 fork workers";;
 stop)
  for p in $(pids); do kill "$p" 2>/dev/null || true; done
  for wid in fork_g1a fork_g1b fork_g1c fork_g1d fork_g2a fork_g2b fork_g2c fork_g2d fork_g3a fork_g3b fork_g3c fork_g3d; do tmux kill-session -t "fork_${wid}" 2>/dev/null || true; done
  echo stopped;;
 status)
  echo "pending=$(count "$ROOT/cases/pending") running=$(count "$ROOT/cases/running") completed=$(count "$ROOT/cases/completed") error=$(count "$ROOT/cases/error") forks=$(count "$ROOT/forks")"
  echo "workers: $(pids | tr '\n' ' ')";;
 *) echo 'usage: start|stop|status' >&2; exit 2;;
esac
