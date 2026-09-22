#!/usr/bin/env bash
set -euo pipefail

WS=/home/leju-suzhou/zjt_ws/token-cd
CKPT=/home/leju-suzhou/zjt_ws/checkpoints/libero/vq-vla-openvla-7b-libero90
STATS="$CKPT/dataset_statistics.json"
ROOT="$WS/artifacts/libero90_openvla_migration_smoke_v1"
UNNORM=libero_90_no_noops
SETTLE=10
MAX_STEPS=400

source "$WS/scripts/activate_libero_openvla.sh"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

TASK_3=KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it
TASK_10=KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet
TASK_49=LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket
TASK_72=LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate
TASK_73=STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy

run_arm() {
  local arm="$1"
  local task="$2"
  local cvd="$3"
  local run_root="$ROOT/runs/$arm"
  local video_root="$ROOT/videos/$arm"
  local log_dir="$ROOT/logs/$arm"
  local log="$log_dir/${task}.log"
  mkdir -p "$run_root" "$video_root" "$log_dir"

  # Keep GPU0 visible only for the MuJoCo EGL context and run the model on the
  # second visible device.  This is the rendering pattern used by the existing
  # Spatial runs and avoids per-physical-GPU EGL framebuffer issues.
  export CUDA_VISIBLE_DEVICES="0,$cvd"
  export MUJOCO_EGL_DEVICE_ID=0
  local model_gpu=1
  {
    echo "[command] arm=$arm task=$task cvd=$cvd $(date -Iseconds)"
    if [[ "$arm" == vanilla ]]; then
      printf '%q ' python "$WS/research/semantic_token_cd/libero_vanilla_rollout.py" \
        --artifact "$run_root" --checkpoint "$CKPT" --suite libero_90 \
        --task "$task" --episodes 0 --gpu "$model_gpu" --unnorm-key "$UNNORM" \
        --dataset-statistics "$STATS" --env-seed 0 --settle-steps "$SETTLE" \
        --max-steps "$MAX_STEPS" --video-dir "$video_root"
      echo
      python "$WS/research/semantic_token_cd/libero_vanilla_rollout.py" \
        --artifact "$run_root" --checkpoint "$CKPT" --suite libero_90 \
        --task "$task" --episodes 0 --gpu "$model_gpu" --unnorm-key "$UNNORM" \
        --dataset-statistics "$STATS" --env-seed 0 --settle-steps "$SETTLE" \
        --max-steps "$MAX_STEPS" --video-dir "$video_root"
    else
      printf '%q ' python "$WS/research/semantic_token_cd/libero_matched_rollout.py" \
        --artifact "$run_root" --checkpoint "$CKPT" --suite libero_90 \
        --task "$task" --episodes 0 --gpu "$model_gpu" --unnorm-key "$UNNORM" \
        --dataset-statistics "$STATS" --entity-mode source_target_libero90 \
        --query-mode instruction_only --attention-layers 11 --attention-heads '' \
        --destination-weight 0.0 --lambda-scale 0.25 --position-mode attention \
        --env-seed 0 --settle-steps "$SETTLE" --max-steps "$MAX_STEPS" \
        --video-dir "$video_root"
      echo
      python "$WS/research/semantic_token_cd/libero_matched_rollout.py" \
        --artifact "$run_root" --checkpoint "$CKPT" --suite libero_90 \
        --task "$task" --episodes 0 --gpu "$model_gpu" --unnorm-key "$UNNORM" \
        --dataset-statistics "$STATS" --entity-mode source_target_libero90 \
        --query-mode instruction_only --attention-layers 11 --attention-heads '' \
        --destination-weight 0.0 --lambda-scale 0.25 --position-mode attention \
        --env-seed 0 --settle-steps "$SETTLE" --max-steps "$MAX_STEPS" \
        --video-dir "$video_root"
    fi
  } >> "$log" 2>&1
}

run_worker() {
  local shard="$1"
  local cvd tasks
  case "$shard" in
    1) cvd=1; tasks=("$TASK_3" "$TASK_10") ;;
    2) cvd=2; tasks=("$TASK_49") ;;
    3) cvd=3; tasks=("$TASK_72" "$TASK_73") ;;
    *) echo "usage: $0 {1|2|3}" >&2; exit 2 ;;
  esac
  mkdir -p "$ROOT/status"
  local marker="$ROOT/status/shard${shard}_COMPLETE"
  : > "$ROOT/status/shard${shard}_RUNNING"
  for task in "${tasks[@]}"; do
    run_arm vanilla "$task" "$cvd"
    run_arm matched "$task" "$cvd"
  done
  mv "$ROOT/status/shard${shard}_RUNNING" "$marker"
}

run_worker "$1"
