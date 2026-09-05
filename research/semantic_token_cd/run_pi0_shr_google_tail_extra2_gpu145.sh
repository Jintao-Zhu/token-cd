#!/usr/bin/env bash
# Second complementary tail wave: one additional model per physical GPU.
set -uo pipefail

cd /home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/miniconda3/envs/pi0/bin/python
ART=artifacts/pi0_shr_5task_0_299_v1
PI0_ROOT=/home/leju-suzhou/zjt_ws/open-pi-zero

export TRANSFORMERS_CACHE=/home/leju-suzhou/zjt_ws/checkpoints/pi0
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PI0_ROOT:/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/SimplerEnv"

mkdir -p "$ART/logs"

run_tail() {
  local gpu=$1 task=$2 seeds=$3 label=$4 status
  local log="$ART/logs/${label}.log"
  while :; do
    env CUDA_VISIBLE_DEVICES="$gpu" "$PY" research/semantic_token_cd/pi0_shr_rollout.py \
      --artifact "$ART" --task "$task" --seeds "$seeds" --gpu "$gpu" \
      --worker-id "$label" >> "$log" 2>&1
    status=$?
    [ "$status" -eq 0 ] && return 0
    printf '%s retry status=%s\n' "$(date --iso-8601=seconds)" "$status" >> "$log"
    sleep 5
  done
}

run_tail 1 google_robot_open_drawer \
  276,279,282,285,288,291,294,297 pi0-extra2-gpu1-open-tail &
run_tail 4 google_robot_open_drawer \
  277,280,283,286,289,292,295,298 pi0-extra2-gpu4-open-tail &
run_tail 5 google_robot_close_drawer \
  278,281,284,287,290,293,296,299 pi0-extra2-gpu5-close-tail &
wait
