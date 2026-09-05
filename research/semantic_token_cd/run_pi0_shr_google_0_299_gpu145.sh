#!/usr/bin/env bash
# Start the Fractal-Beta Google Robot portion of PI0_SHR_5TASK_0_299_V1.
# WidowX remains in the chained five-task launcher because it requires the
# independently downloaded Bridge-Beta checkpoint.
set -uo pipefail

cd /home/leju-suzhou/zjt_ws/token-cd

PY=/home/leju-suzhou/miniconda3/envs/pi0/bin/python
ART=artifacts/pi0_shr_5task_0_299_v1
PI0_ROOT=/home/leju-suzhou/zjt_ws/open-pi-zero
FRACTAL=/home/leju-suzhou/zjt_ws/checkpoints/pi0/fractal_beta_step29576_2024-12-29_13-10_42.pt
FRACTAL_SIZE=11774183772
# Equal worker counts avoid leaving pick_coke_can as the long tail.  Each
# physical GPU receives four models (12 workers total across GPUs 1, 4, 5).
TASKS=(google_robot_close_drawer google_robot_open_drawer google_robot_move_near google_robot_pick_coke_can)
WORKERS=(3 3 3 3)
GPUS=(1 4 5)
ARMS=(pi0_vanilla pi0_shr_harmonic)

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

mkdir -p "$ART/logs" "$ART/episodes"

preflight() {
  [ -x "$PY" ] || { echo "missing Pi0 Python: $PY" >&2; return 1; }
  [ -s "$FRACTAL" ] || { echo "missing Fractal checkpoint: $FRACTAL" >&2; return 1; }
  [ "$(stat -c %s "$FRACTAL")" -eq "$FRACTAL_SIZE" ] || {
    echo "Fractal checkpoint size mismatch" >&2; return 1;
  }
  "$PY" -c 'import torch, hydra, sklearn, simpler_env; from src.model.vla.pizero import PiZeroInference; print("Pi0 Google preflight OK", torch.__version__)'
}

seed_spec() {
  local worker=$1 nworkers=$2 seed spec=""
  for ((seed=worker; seed<300; seed+=nworkers)); do
    [ -z "$spec" ] || spec+=","
    spec+="$seed"
  done
  printf '%s' "$spec"
}

completed_in_spec() {
  local task=$1 spec=$2 arm seed count=0 old_ifs=$IFS
  IFS=,
  read -ra seeds <<< "$spec"
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
  expected=$((2 * (300 / nworkers + (worker < 300 % nworkers ? 1 : 0))))
  log="$ART/logs/pi0_google_${task}_${worker}of${nworkers}.log"
  lock="$ART/logs/pi0_google_${task}_${worker}of${nworkers}.lock"
  (
    flock -n 9 || return 0
    while :; do
      count=$(completed_in_spec "$task" "$spec")
      if [ "$count" -eq "$expected" ]; then
        touch "$log.done"
        return 0
      fi
      printf '%s gpu=%s complete=%s/%s\n' "$(date --iso-8601=seconds)" "$gpu" "$count" "$expected" >> "$log"
      env CUDA_VISIBLE_DEVICES="$gpu" "$PY" research/semantic_token_cd/pi0_shr_rollout.py \
        --artifact "$ART" --task "$task" --seeds "$spec" --gpu "$gpu" \
        --worker-id "pi0-google-${task}-${worker}of${nworkers}" >> "$log" 2>&1
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
for task_index in 0 1 2 3; do
  task=${TASKS[$task_index]}
  nworkers=${WORKERS[$task_index]}
  for ((worker=0; worker<nworkers; worker++)); do
    gpu=${GPUS[$((job % ${#GPUS[@]}))]}
    run_worker "$task" "$nworkers" "$worker" "$gpu" &
    job=$((job + 1))
  done
done
wait
printf 'Pi0-SHR Google portion complete: 2400/2400\n' > "$ART/logs/GOOGLE_COMPLETE"
