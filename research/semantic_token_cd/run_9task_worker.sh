#!/bin/bash
# 9-task benchmark worker: reference (3-arm) then dual, for two tasks sequentially.
# Usage: run_9task_worker.sh <worker_id> <task1> <task2>
# All three workers share GPU 0. Each task's dual stage requires its reference
# pairing_manifest, so the two stages run sequentially within each task.
set -uo pipefail

WORKER_ID="$1"
TASK1="$2"
TASK2="$3"
GPU=0

REPO_ROOT="/data/docker/dev_zjt/data/code"
ART_REF="$REPO_ROOT/artifacts/uniform_vs_semantic_attention_cd_450_v1"
ART_DUAL="$REPO_ROOT/artifacts/orthogonal_dual_attention_cd_50_v1"
VANILLA_SRC="$REPO_ROOT/artifacts/simpler_distractor_semantic_entity_cd_phase0_v2"
REUSE_SRC="$REPO_ROOT/artifacts/spatial_grid_vs_random_attention_cd_v1"
VENV="$REPO_ROOT/task1/.venvs/openvla-ar/bin/python"
PYTHONPATH="$REPO_ROOT/task1/shim_site:$REPO_ROOT:$REPO_ROOT/official-reproductions/pcd_openvla_simpler_box_31b027e/source/PCD"
LOGDIR="$ART_DUAL/worker_logs"
mkdir -p "$LOGDIR"
cd "$REPO_ROOT" || exit 1

run_stage() {
  local stage="$1" task="$2" label="$3"
  echo "=== [worker $WORKER_ID] [$label] $task : $stage START $(date '+%F %H:%M:%S') PID=$$ GPU=$GPU ==="
  if [ "$stage" = "reference" ]; then
    env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" TF_CPP_MIN_LOG_LEVEL=3 \
      PYTHONPATH="$PYTHONPATH" \
      "$VENV" research/semantic_token_cd/uniform_semantic_rollout.py \
        --artifact "$ART_REF" \
        --vanilla-artifact "$VANILLA_SRC" \
        --reuse-artifact "$REUSE_SRC" \
        --task "$task" --gpu "$GPU"
  else
    env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" TF_CPP_MIN_LOG_LEVEL=3 \
      PYTHONPATH="$PYTHONPATH" \
      "$VENV" research/semantic_token_cd/orthogonal_dual_experiment.py \
        --artifact "$ART_DUAL" \
        --reference-artifact "$ART_REF" \
        --task "$task" --gpu "$GPU"
  fi
  local rc=$?
  echo "=== [worker $WORKER_ID] [$label] $task : $stage DONE rc=$rc $(date '+%F %H:%M:%S') ==="
  return $rc
}

run_task() {
  local task="$1" label="$2"
  run_stage reference "$task" "$label" || { echo "ABORT worker $WORKER_ID after reference $task"; return 1; }
  run_stage dual "$task" "$label" || { echo "ABORT worker $WORKER_ID after dual $task"; return 1; }
}

run_task "$TASK1" "A" || exit 1
run_task "$TASK2" "B" || exit 1
echo "=== [worker $WORKER_ID] ALL DONE $(date '+%F %H:%M:%S') ==="
