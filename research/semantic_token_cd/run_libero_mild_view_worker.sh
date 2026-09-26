#!/usr/bin/env bash
set -euo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
OUT="$WS/artifacts/libero_transfer_diagnostics/libero_mild_view_logits"
worker_gpu="${1:?usage: run_libero_mild_view_worker.sh GPU_ID}"
worker_attempt="${2:-initial}"
case "$worker_gpu" in 1|2|3|6|7) ;; *) echo "GPU not authorized: $worker_gpu" >&2; exit 2 ;; esac
worker_visible="$worker_gpu"
worker_render="$worker_gpu"
worker_device_index=0
worker_log_gpu="$worker_gpu"
if [ "$worker_gpu" = 2 ]; then
  # GPU 2 failed EGL FBO creation when used as renderer. Match the established
  # LIBERO runner mapping: infer on physical GPU 2, render through GPU 1.
  worker_visible=1,2
  worker_render=1
  worker_device_index=1
  worker_log_gpu=2_retry
fi
if [ "$worker_gpu" = 3 ]; then
  # GPU 3 completed the first case but exited while resetting its EGL context.
  # Reuse the stable shared renderer mapping used by the official LIBERO jobs.
  worker_visible=1,3
  worker_render=1
  worker_device_index=1
  worker_log_gpu=3_retry
fi
cd "$WS"
source scripts/activate_libero_openvla.sh
export CUDA_VISIBLE_DEVICES="$worker_visible"
export MUJOCO_EGL_DEVICE_ID="$worker_render"
export TOKENIZERS_PARALLELISM=false
worker_log="$OUT/logs/gpu${worker_log_gpu}.log"
if [ "$worker_attempt" != initial ]; then
  worker_log="$OUT/logs/gpu${worker_log_gpu}_attempt_${worker_attempt}.log"
fi
finish_worker() {
  local worker_status=$?
  echo "$worker_status" > "$OUT/logs/gpu${worker_log_gpu}_${worker_attempt}.exit"
  echo "worker_exit=$worker_status" >> "$worker_log"
}
trap finish_worker EXIT
python research/semantic_token_cd/collect_libero_mild_view_stability.py \
  --gpu "$worker_gpu" \
  --device-index "$worker_device_index" \
  --case-list "$OUT/manifests/gpu${worker_gpu}.txt" \
  --skip-existing \
  --rgb-source live-replay \
  --out "$OUT" \
  > "$worker_log" 2>&1
