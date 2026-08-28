#!/usr/bin/env bash
# SIMPLER_DISTRACTOR_SEMANTIC_ENTITY_CD_PHASE0_V2: 3 tasks x 50 seeds x 3 arms.
set -euo pipefail
cd /data/docker/dev_zjt/data/code

export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH="task1/shim_site:$PWD:official-reproductions/pcd_openvla_simpler_box_31b027e/source/PCD"

PY=task1/.venvs/openvla-ar/bin/python
ART=artifacts/simpler_distractor_semantic_entity_cd_phase0_v2
LOG_DIR="$ART/rollout_logs"
SEEDS=$(seq -s, 0 49)
mkdir -p "$LOG_DIR"

run_task() {
  local task="$1"
  "$PY" -u research/semantic_token_cd/distractor_rollout.py \
    --artifact "$ART" --task "$task" --seeds "$SEEDS" --gpu 0 \
    > "$LOG_DIR/${task}.log" 2>&1
  echo "done: $task"
}
export PY ART LOG_DIR SEEDS
export -f run_task

printf '%s\n' \
  google_robot_open_drawer \
  google_robot_close_drawer \
  google_robot_pick_coke_can \
  | xargs -P 3 -n 1 bash -c 'run_task "$1"' _

"$PY" research/semantic_token_cd/analyze_distractor_v2.py --artifact "$ART"
