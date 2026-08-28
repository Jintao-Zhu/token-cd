#!/usr/bin/env bash
# SEMANTIC_ENTITY_CD_ROLLOUT_PILOT_V1 — full run driver (9 tasks × 10 seeds × 3 arms).
# Resumable: rollout_pilot.py skips existing *_summary.json files.
set -u
cd /data/docker/dev_zjt/data/code

LOG_DIR="${CLAUDE_JOB_DIR:-/home/zjt/.claude/jobs/5090eadf}/tmp"
mkdir -p "$LOG_DIR"

export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="task1/shim_site:$PWD:official-reproductions/pcd_openvla_simpler_box_31b027e/source/PCD"

PY=task1/.venvs/openvla-ar/bin/python
ART=artifacts/semantic_entity_cd_rollout_pilot_v1
SEEDS="0,1,2,3,4,5,6,7,8,9"

TASKS=(
  google_robot_close_drawer
  google_robot_move_near
  google_robot_open_drawer
  google_robot_pick_coke_can
  google_robot_place_apple_in_closed_top_drawer
  widowx_carrot_on_plate
  widowx_put_eggplant_in_basket
  widowx_spoon_on_towel
  widowx_stack_cube
)

for TASK in "${TASKS[@]}"; do
  LOG="$LOG_DIR/rollout_${TASK}.log"
  echo "[$(date '+%F %T')] START $TASK" | tee -a "$LOG_DIR/rollout_master.log"
  if "$PY" research/semantic_token_cd/rollout_pilot.py \
        --artifact "$ART" --task "$TASK" --seeds "$SEEDS" >> "$LOG" 2>&1; then
    echo "[$(date '+%F %T')] DONE  $TASK (exit 0)" | tee -a "$LOG_DIR/rollout_master.log"
  else
    echo "[$(date '+%F %T')] FAIL  $TASK (exit $?)" | tee -a "$LOG_DIR/rollout_master.log"
  fi
done
echo "[$(date '+%F %T')] ALL_TASKS_COMPLETE" | tee -a "$LOG_DIR/rollout_master.log"
