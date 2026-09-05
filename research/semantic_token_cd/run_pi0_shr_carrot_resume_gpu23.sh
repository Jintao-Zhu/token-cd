#!/usr/bin/env bash
set -uo pipefail

cd /home/leju-suzhou/zjt_ws/token-cd

PY=/home/leju-suzhou/miniconda3/envs/pi0/bin/python
ART=artifacts/pi0_shr_5task_0_299_v1
TASK=widowx_carrot_on_plate
PI0_ROOT=/home/leju-suzhou/zjt_ws/open-pi-zero
GPUS=(2 3)
NWORKERS=4
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

mkdir -p "$ART/logs" "$ART/episodes/$TASK"

seed_spec() {
  local worker=$1 seed spec=""
  for ((seed=worker; seed<300; seed+=NWORKERS)); do
    [ -z "$spec" ] || spec+=","
    spec+="$seed"
  done
  printf '%s' "$spec"
}

completed_spec() {
  local spec=$1 arm seed count=0 old_ifs=$IFS
  IFS=,
  read -ra seeds <<< "$spec"
  IFS=$old_ifs
  for arm in "${ARMS[@]}"; do
    for seed in "${seeds[@]}"; do
      local stem="$ART/episodes/$TASK/$arm/episode_$(printf '%03d' "$seed")"
      if [ -f "${stem}_summary.json" ] && [ -f "${stem}_arrays.npz" ]; then
        count=$((count + 1))
      fi
    done
  done
  printf '%s' "$count"
}

run_worker() {
  local worker=$1 gpu=$2 spec expected log lock count status
  spec=$(seed_spec "$worker")
  expected=$((2 * (300 / NWORKERS + (worker < 300 % NWORKERS ? 1 : 0))))
  log="$ART/logs/${TASK}_gpu23_${worker}of${NWORKERS}.log"
  lock="$ART/logs/${TASK}_gpu23_${worker}of${NWORKERS}.lock"
  (
    flock -n 9 || exit 0
    while :; do
      count=$(completed_spec "$spec")
      if [ "$count" -eq "$expected" ]; then
        printf '%s complete=%s/%s\n' "$(date --iso-8601=seconds)" "$count" "$expected" >> "$log"
        exit 0
      fi
      printf '%s gpu=%s complete=%s/%s\n' \
        "$(date --iso-8601=seconds)" "$gpu" "$count" "$expected" >> "$log"
      env CUDA_VISIBLE_DEVICES="$gpu" "$PY" \
        research/semantic_token_cd/pi0_shr_rollout.py \
        --artifact "$ART" --task "$TASK" --seeds "$spec" --gpu "$gpu" \
        --worker-id "${TASK}-gpu23-${worker}of${NWORKERS}" >> "$log" 2>&1
      status=$?
      count=$(completed_spec "$spec")
      printf '%s exit=%s complete=%s/%s; retrying\n' \
        "$(date --iso-8601=seconds)" "$status" "$count" "$expected" >> "$log"
      sleep 5
    done
  ) 9> "$lock"
}

for ((worker=0; worker<NWORKERS; worker++)); do
  gpu=${GPUS[$((worker % ${#GPUS[@]}))]}
  run_worker "$worker" "$gpu" &
done
wait

total_count=$(find "$ART/episodes" -type f -name 'episode_*_summary.json' | wc -l)
if [ "$total_count" -eq 3000 ]; then
  "$PY" research/semantic_token_cd/analyze_pi0_shr.py --artifact "$ART" \
    > "$ART/logs/final_statistics.txt"
  printf 'Pi0-SHR experiment complete: 3000/3000\n' > "$ART/logs/COMPLETE"
else
  printf 'Pi0-SHR workers ended with %s/3000 summaries\n' "$total_count" >&2
  exit 1
fi
