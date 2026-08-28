#!/bin/bash
# Usage: run_four_arm_6task_worker.sh <worker_id> <task1> <task2>
set -euo pipefail

WORKER_ID="$1"
TASK1="$2"
TASK2="$3"
GPU=0

REPO_ROOT="/data/docker/dev_zjt/data/code"
ARTIFACT="$REPO_ROOT/artifacts/orthogonal_dual_four_arm_6task_50_v1"
VENV="$REPO_ROOT/task1/.venvs/openvla-ar/bin/python"
PCD_SOURCE="$REPO_ROOT/official-reproductions/pcd_openvla_simpler_box_31b027e/source/PCD"
PYTHONPATH_VALUE="$REPO_ROOT/task1/shim_site:$REPO_ROOT:$PCD_SOURCE"

cd "$REPO_ROOT"

run_task() {
  local task="$1"
  echo "=== worker=$WORKER_ID task=$task START $(date '+%F %T') ==="
  env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" TF_CPP_MIN_LOG_LEVEL=3 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
    PYTHONPATH="$PYTHONPATH_VALUE" \
    "$VENV" research/semantic_token_cd/four_arm_orthogonal_rollout.py \
      --artifact "$ARTIFACT" --task "$task" --gpu "$GPU"
  echo "=== worker=$WORKER_ID task=$task DONE $(date '+%F %T') ==="
}

run_task "$TASK1"
run_task "$TASK2"
echo "=== worker=$WORKER_ID ALL_DONE $(date '+%F %T') ==="
