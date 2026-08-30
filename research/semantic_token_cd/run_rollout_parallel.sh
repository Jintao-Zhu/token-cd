#!/usr/bin/env bash
# SEMANTIC_ENTITY_CD_ROLLOUT_PILOT_V1 — parallel driver (3 concurrent tasks).
# Each task is independent (own model + env + summary dir); rollout_pilot.py
# skips existing *_summary.json, so this is fully resumable.
# Slowest tasks are listed first so xargs -P 3 starts them immediately and the
# fast tasks backfill behind them.
set -u
cd /home/leju-suzhou/zjt_ws/token-cd

LOG_DIR="${CLAUDE_JOB_DIR:-/home/zjt/.claude/jobs/5090eadf}/tmp"
mkdir -p "$LOG_DIR"

export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false
# Pin numpy/sklearn/sapien CPU threads — avoid multi-process thrash on the shared box.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH="task1/shim_site:$PWD:official-reproductions/pcd_openvla_simpler_box_31b027e/source/PCD"

PY=task1/.venvs/openvla-ar/bin/python
ART=artifacts/semantic_entity_cd_rollout_pilot_v1
SEEDS="0,1,2,3,4,5,6,7,8,9"
PARALLEL=3

TASKS=(
  google_robot_close_drawer
  google_robot_open_drawer
  google_robot_place_apple_in_closed_top_drawer
  google_robot_move_near
  google_robot_pick_coke_can
  widowx_carrot_on_plate
  widowx_put_eggplant_in_basket
  widowx_spoon_on_towel
  widowx_stack_cube
)

export LOG_DIR PY ART SEEDS

run_task() {
  local TASK="$1"
  local LOG="$LOG_DIR/rollout_${TASK}.log"
  echo "[$(date '+%F %T')] START $TASK" >> "$LOG_DIR/rollout_master.log"
  if "$PY" research/semantic_token_cd/rollout_pilot.py \
        --artifact "$ART" --task "$TASK" --seeds "$SEEDS" >> "$LOG" 2>&1; then
    echo "[$(date '+%F %T')] DONE  $TASK (exit 0)" >> "$LOG_DIR/rollout_master.log"
  else
    local rc=$?
    echo "[$(date '+%F %T')] FAIL  $TASK (exit $rc)" >> "$LOG_DIR/rollout_master.log"
  fi
}
export -f run_task

printf '%s\n' "${TASKS[@]}" | xargs -P "$PARALLEL" -n 1 bash -c 'run_task "$1"' _

echo "[$(date '+%F %T')] ALL_TASKS_COMPLETE" >> "$LOG_DIR/rollout_master.log"
