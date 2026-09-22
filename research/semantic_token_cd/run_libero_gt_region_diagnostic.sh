#!/usr/bin/env bash
set -uo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/libero-openvla-env/bin/python
CKPT=/home/leju-suzhou/zjt_ws/checkpoints/libero/openvla-7b-finetuned-libero-spatial
ROOT="$WS/artifacts/libero_gt_region_diagnostic_v2"
source "$WS/scripts/activate_libero_openvla.sh"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs"
T2=pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate
T5=pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate
T8=pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate
T10=pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate
TASKS=("$T2" "$T5" "$T8" "$T10")

run_worker() {
  local arm="$1" query="$2" layers="$3" heads="$4" scale="$5" position="$6" states="$7" cvd="$8" worker="$9"
  local arm_root="$ROOT/$arm"; mkdir -p "$arm_root"
  export CUDA_VISIBLE_DEVICES="$cvd" MUJOCO_EGL_DEVICE_ID=0
  for task in "${TASKS[@]}"; do
    echo "[worker $worker] arm=$arm task=$task states=$states $(date '+%F %T')"
    local attempt=1
    while true; do
      if "$PY" research/semantic_token_cd/libero_matched_rollout.py \
        --artifact "$arm_root" --checkpoint "$CKPT" --task "$task" \
        --episodes "$states" --gpu 1 --entity-mode source_target \
        --query-mode "$query" --attention-layers "$layers" --attention-heads "$heads" \
        --destination-weight 0.0 --lambda-scale "$scale" --position-mode "$position" \
        --env-seed 0 --settle-steps 10 --max-steps 220; then break; fi
      if [[ "$attempt" -ge 3 ]]; then return 1; fi
      echo "[worker $worker] retry $attempt/3 arm=$arm task=$task $(date '+%F %T')" >&2
      attempt=$((attempt+1)); sleep 60
    done
  done
}

pids=()
# Batch 1: GT target + missing 0-19 baselines.
run_worker gt_target_lam125 instruction_only 11 '' 0.25 gt_target 0-16 0,1 gt_t1_l125 > "$ROOT/logs/gt_target_lam125_t1.log" 2>&1 & pids+=("$!"); sleep 4
run_worker gt_target_lam125 instruction_only 11 '' 0.25 gt_target 17-33 0,2 gt_t2_l125 > "$ROOT/logs/gt_target_lam125_t2.log" 2>&1 & pids+=("$!"); sleep 4
run_worker gt_target_lam125 instruction_only 11 '' 0.25 gt_target 34-49 0,3 gt_t3_l125 > "$ROOT/logs/gt_target_lam125_t3.log" 2>&1 & pids+=("$!"); sleep 4
run_worker gt_target_lam500 instruction_only 11 '' 1.0 gt_target 0-16 0,1 gt_t1_l500 > "$ROOT/logs/gt_target_lam500_t1.log" 2>&1 & pids+=("$!"); sleep 4
run_worker gt_target_lam500 instruction_only 11 '' 1.0 gt_target 17-33 0,2 gt_t2_l500 > "$ROOT/logs/gt_target_lam500_t2.log" 2>&1 & pids+=("$!"); sleep 4
run_worker gt_target_lam500 instruction_only 11 '' 1.0 gt_target 34-49 0,3 gt_t3_l500 > "$ROOT/logs/gt_target_lam500_t3.log" 2>&1 & pids+=("$!"); sleep 4
run_worker current_lam125_extra instruction_only 11 '' 0.25 attention 0-19 0,1 cur125_extra > "$ROOT/logs/current_lam125_extra.log" 2>&1 & pids+=("$!"); sleep 4
run_worker selected5_lam500_extra target_relation_endpoint 11 '14:5,7:14,9:12,19:11,21:2' 1.0 attention 0-19 0,2 sel5_500_extra > "$ROOT/logs/selected5_lam500_extra.log" 2>&1 & pids+=("$!"); sleep 4
run_worker selected5_lam125_extra target_relation_endpoint 11 '14:5,7:14,9:12,19:11,21:2' 0.25 attention 0-19 0,3 sel5_125_extra > "$ROOT/logs/selected5_lam125_extra.log" 2>&1 & pids+=("$!")
st=(); for p in "${pids[@]}"; do if wait "$p"; then st+=(0); else st+=("$?"); fi; done
for s in "${st[@]}"; do [[ "$s" -eq 0 ]] || { echo "batch1 statuses ${st[*]}" >&2; exit 1; }; done

pids=()
# Batch 2: GT target + reference.
run_worker gt_target_reference_lam125 instruction_only 11 '' 0.25 gt_target_reference 0-16 0,1 gtr_t1_l125 > "$ROOT/logs/gt_target_reference_lam125_t1.log" 2>&1 & pids+=("$!"); sleep 4
run_worker gt_target_reference_lam125 instruction_only 11 '' 0.25 gt_target_reference 17-33 0,2 gtr_t2_l125 > "$ROOT/logs/gt_target_reference_lam125_t2.log" 2>&1 & pids+=("$!"); sleep 4
run_worker gt_target_reference_lam125 instruction_only 11 '' 0.25 gt_target_reference 34-49 0,3 gtr_t3_l125 > "$ROOT/logs/gt_target_reference_lam125_t3.log" 2>&1 & pids+=("$!"); sleep 4
run_worker gt_target_reference_lam500 instruction_only 11 '' 1.0 gt_target_reference 0-16 0,1 gtr_t1_l500 > "$ROOT/logs/gt_target_reference_lam500_t1.log" 2>&1 & pids+=("$!"); sleep 4
run_worker gt_target_reference_lam500 instruction_only 11 '' 1.0 gt_target_reference 17-33 0,2 gtr_t2_l500 > "$ROOT/logs/gt_target_reference_lam500_t2.log" 2>&1 & pids+=("$!"); sleep 4
run_worker gt_target_reference_lam500 instruction_only 11 '' 1.0 gt_target_reference 34-49 0,3 gtr_t3_l500 > "$ROOT/logs/gt_target_reference_lam500_t3.log" 2>&1 & pids+=("$!")
st=(); for p in "${pids[@]}"; do if wait "$p"; then st+=(0); else st+=("$?"); fi; done
for s in "${st[@]}"; do [[ "$s" -eq 0 ]] || { echo "batch2 statuses ${st[*]}" >&2; exit 1; }; done
echo "GT_REGION_DIAGNOSTIC_DONE $(date '+%F %T')"
