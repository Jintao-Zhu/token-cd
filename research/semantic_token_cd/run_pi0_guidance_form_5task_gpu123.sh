#!/usr/bin/env bash
# Phase 1: paired Pi0 guidance-form search.
#
# Design follows the Phase-0 diagnosis (dense+tiny vs sparse+large) and the
# per-task paired re-analysis of PI0_SHR_5TASK_0_299_V1:
#   pick_coke_can is excluded because the vanilla policy already scores 98.7%
#   (296/300) there, so the task cannot register an improvement; the remaining
#   four tasks span 47%-85% vanilla success.
#
# 4 tasks x 100 seeds x 8 arms = 3200 episodes.  GPUs 1,2,3 only.
set -uo pipefail

cd /home/leju-suzhou/zjt_ws/token-cd

PY=/home/leju-suzhou/miniconda3/envs/pi0/bin/python
ART=artifacts/pi0_guidance_form_5task_100seed_v1
REF_SNAP=artifacts/pi0_shr_5task_0_299_v1/snapshots
PI0_ROOT=/home/leju-suzhou/zjt_ws/open-pi-zero
FRACTAL=/home/leju-suzhou/zjt_ws/checkpoints/pi0/fractal_beta_step29576_2024-12-29_13-10_42.pt
BRIDGE=/home/leju-suzhou/zjt_ws/checkpoints/pi0/bridge_beta_step19296_2024-12-26_22-30_42.pt
FRACTAL_SIZE=11774183772

# Tasks with measurable headroom (vanilla success in [20%, 90%]), workers
# proportional to measured single-model cost.
TASKS=(google_robot_close_drawer google_robot_open_drawer google_robot_move_near widowx_carrot_on_plate)
WORKERS=(4 4 2 2)
GPUS=(1 2 3)
ARMS=(pi0_vanilla pi0_dense_l050 pi0_dense_l150 pi0_dense_l300 \
      pi0_top2_l050 pi0_top2_l150 pi0_last3_l050 pi0_relgate015_l050)
NSEEDS=100
NARMS=${#ARMS[@]}

export TRANSFORMERS_CACHE=/home/leju-suzhou/zjt_ws/checkpoints/pi0
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export PYTHONPATH="$PI0_ROOT:/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/SimplerEnv"

mkdir -p "$ART/logs" "$ART/episodes" "$ART/snapshots"

# Reuse the exact simulator snapshots already used by the reference protocol so
# the initial states are identical across the two studies.
for task in "${TASKS[@]}"; do
  if [ -d "$REF_SNAP/$task" ] && [ ! -e "$ART/snapshots/$task" ]; then
    ln -s "$(pwd)/$REF_SNAP/$task" "$ART/snapshots/$task"
  fi
done

preflight() {
  [ -x "$PY" ] || { echo "missing Pi0 python: $PY" >&2; return 1; }
  for ck in "$FRACTAL" "$BRIDGE"; do
    [ -s "$ck" ] || { echo "missing checkpoint: $ck" >&2; return 1; }
  done
  [ "$(stat -c %s "$FRACTAL")" -eq "$FRACTAL_SIZE" ] || {
    echo "Fractal checkpoint size mismatch" >&2; return 1;
  }
  "$PY" -c 'import torch, hydra, sklearn, simpler_env; from src.model.vla.pizero import PiZeroInference; print("variant preflight OK", torch.__version__)'
}

seed_spec() {
  local worker=$1 nworkers=$2 spec=""
  for ((seed=worker; seed<NSEEDS; seed+=nworkers)); do
    [ -z "$spec" ] || spec+=","
    spec+="$seed"
  done
  printf '%s' "$spec"
}

completed_in_spec() {
  local task=$1 spec=$2 arm seed count=0 old_ifs=$IFS
  IFS=, read -ra seeds <<< "$spec"
  IFS=$old_ifs
  for arm in "${ARMS[@]}"; do
    for seed in "${seeds[@]}"; do
      if [ -f "$ART/episodes/$task/$arm/episode_$(printf '%03d' "$seed")_summary.json" ] && \
         [ -f "$ART/episodes/$task/$arm/episode_$(printf '%03d' "$seed")_arrays.npz" ]; then
        count=$((count + 1))
      fi
    done
  done
  printf '%s' "$count"
}

run_worker() {
  local task=$1 nworkers=$2 worker=$3 gpu=$4 spec expected log lock count status
  spec=$(seed_spec "$worker" "$nworkers")
  if [ "$worker" -lt $((NSEEDS % nworkers)) ]; then
    expected=$((NARMS * (NSEEDS / nworkers + 1)))
  else
    expected=$((NARMS * (NSEEDS / nworkers)))
  fi
  log="$ART/logs/${task}_${worker}of${nworkers}.log"
  lock="$ART/logs/${task}_${worker}of${nworkers}.lock"
  (
    flock -n 9 || return 0
    while :; do
      count=$(completed_in_spec "$task" "$spec")
      if [ "$count" -eq "$expected" ]; then
        touch "$log.done"
        return 0
      fi
      printf '%s gpu=%s complete=%s/%s\n' "$(date --iso-8601=seconds)" "$gpu" "$count" "$expected" >> "$log"
      env CUDA_VISIBLE_DEVICES="$gpu" "$PY" research/semantic_token_cd/pi0_variant_rollout.py \
        --artifact "$ART" --task "$task" --seeds "$spec" --gpu "$gpu" \
        --arms "$(IFS=,; echo "${ARMS[*]}")" \
        --worker-id "variant-${task}-${worker}of${nworkers}" >> "$log" 2>&1
      status=$?
      count=$(completed_in_spec "$task" "$spec")
      [ "$count" -eq "$expected" ] && continue
      printf '%s retry status=%s complete=%s/%s\n' "$(date --iso-8601=seconds)" "$status" "$count" "$expected" >> "$log"
      sleep 5
    done
  ) 9> "$lock"
}

preflight || exit 1
job=0
for task_index in "${!TASKS[@]}"; do
  task=${TASKS[$task_index]}
  nworkers=${WORKERS[$task_index]}
  for ((worker=0; worker<nworkers; worker++)); do
    gpu=${GPUS[$((job % ${#GPUS[@]}))]}
    run_worker "$task" "$nworkers" "$worker" "$gpu" &
    job=$((job + 1))
  done
done
wait
total=$((NSEEDS * NARMS * ${#TASKS[@]}))
printf 'Pi0 guidance-form search complete: %s episodes\n' "$total" > "$ART/logs/COMPLETE"
