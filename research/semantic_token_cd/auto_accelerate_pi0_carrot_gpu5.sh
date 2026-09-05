#!/usr/bin/env bash
set -uo pipefail

cd /home/leju-suzhou/zjt_ws/token-cd

PY=/home/leju-suzhou/miniconda3/envs/pi0/bin/python
ART=artifacts/pi0_shr_5task_0_299_v1
TASK=widowx_carrot_on_plate
GPU=5
NWORKERS=1
PI0_ROOT=/home/leju-suzhou/zjt_ws/open-pi-zero
WATCH_LOG="$ART/logs/carrot_gpu5_accelerator.log"
ORIGINAL_LOCK="$ART/logs/widowx_carrot_on_plate_0of1.lock"

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

mkdir -p "$ART/logs"

completed_task() {
  local task=$1 arm seed count=0
  for arm in pi0_vanilla pi0_shr_harmonic; do
    for seed in $(seq 0 299); do
      if [ -f "$ART/episodes/$task/$arm/episode_$(printf '%03d' "$seed")_summary.json" ] && \
         [ -f "$ART/episodes/$task/$arm/episode_$(printf '%03d' "$seed")_arrays.npz" ]; then
        count=$((count + 1))
      fi
    done
  done
  printf '%s' "$count"
}

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
  for arm in pi0_vanilla pi0_shr_harmonic; do
    for seed in "${seeds[@]}"; do
      if [ -f "$ART/episodes/$TASK/$arm/episode_$(printf '%03d' "$seed")_summary.json" ] && \
         [ -f "$ART/episodes/$TASK/$arm/episode_$(printf '%03d' "$seed")_arrays.npz" ]; then
        count=$((count + 1))
      fi
    done
  done
  printf '%s' "$count"
}

stop_original_worker() {
  local task=$1 worker_id=$2 lock=$3 pid cmd
  while read -r pid; do
    [ -n "$pid" ] || continue
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    if [[ "$cmd" == *"pi0_shr_rollout.py"*"--task $task"*"--worker-id $worker_id"* ]]; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done < <(pgrep -f "^$PY .*pi0_shr_rollout.py.*--task $task" || true)

  sleep 2
  for pid in $(fuser "$lock" 2>/dev/null || true); do
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    if [[ "$cmd" == *"run_pi0_shr_5task_0_299_gpu145.sh"* ]]; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  sleep 3
}

finish_close_tail() {
  local count status log="$ART/logs/google_robot_close_drawer_accelerated_tail.log"
  while :; do
    count=$(completed_task google_robot_close_drawer)
    [ "$count" -eq 600 ] && return 0
    printf '%s gpu=%s close_drawer=%s/600\n' \
      "$(date --iso-8601=seconds)" "$GPU" "$count" >> "$log"
    env CUDA_VISIBLE_DEVICES="$GPU" "$PY" \
      research/semantic_token_cd/pi0_shr_rollout.py \
      --artifact "$ART" --task google_robot_close_drawer --seeds 258,270 \
      --gpu "$GPU" --worker-id google_robot_close_drawer-accelerated-tail \
      >> "$log" 2>&1
    status=$?
    printf '%s exit=%s close_drawer=%s/600; retrying\n' \
      "$(date --iso-8601=seconds)" "$status" \
      "$(completed_task google_robot_close_drawer)" >> "$log"
    sleep 5
  done
}

run_shard() {
  local worker=$1 spec expected=$((600 / NWORKERS)) count status log lock
  spec=$(seed_spec "$worker")
  log="$ART/logs/${TASK}_accelerated_${worker}of${NWORKERS}.log"
  lock="$ART/logs/${TASK}_accelerated_${worker}of${NWORKERS}.lock"
  (
    flock -n 9 || exit 0
    while :; do
      count=$(completed_spec "$spec")
      if [ "$count" -eq "$expected" ]; then
        printf '%s complete=%s/%s\n' "$(date --iso-8601=seconds)" "$count" "$expected" >> "$log"
        exit 0
      fi
      printf '%s gpu=%s complete=%s/%s\n' \
        "$(date --iso-8601=seconds)" "$GPU" "$count" "$expected" >> "$log"
      env CUDA_VISIBLE_DEVICES="$GPU" "$PY" \
        research/semantic_token_cd/pi0_shr_rollout.py \
        --artifact "$ART" --task "$TASK" --seeds "$spec" --gpu "$GPU" \
        --worker-id "${TASK}-accelerated-${worker}of${NWORKERS}" >> "$log" 2>&1
      status=$?
      count=$(completed_spec "$spec")
      printf '%s exit=%s complete=%s/%s; retrying\n' \
        "$(date --iso-8601=seconds)" "$status" "$count" "$expected" >> "$log"
      sleep 5
    done
  ) 9> "$lock"
}

printf '%s taking over stalled close tail and single carrot worker\n' \
  "$(date --iso-8601=seconds)" >> "$WATCH_LOG"
stop_original_worker google_robot_close_drawer google_robot_close_drawer-2of4 \
  "$ART/logs/google_robot_close_drawer_2of4.lock"
stop_original_worker "$TASK" "$TASK-0of1" "$ORIGINAL_LOCK"
finish_close_tail
printf '%s close_drawer complete; starting four carrot shards\n' \
  "$(date --iso-8601=seconds)" >> "$WATCH_LOG"

for ((worker=0; worker<NWORKERS; worker++)); do
  run_shard "$worker" &
done
wait

carrot_count=$(completed_task "$TASK")
total_count=$(find "$ART/episodes" -type f -name 'episode_*_summary.json' | wc -l)
if [ "$carrot_count" -eq 600 ] && [ "$total_count" -eq 3000 ]; then
  "$PY" research/semantic_token_cd/analyze_pi0_shr.py --artifact "$ART" \
    > "$ART/logs/final_statistics.txt"
  printf 'Pi0-SHR experiment complete: 3000/3000\n' > "$ART/logs/COMPLETE"
  printf '%s COMPLETE 3000/3000\n' "$(date --iso-8601=seconds)" >> "$WATCH_LOG"
else
  printf '%s ERROR carrot=%s/600 total=%s/3000\n' \
    "$(date --iso-8601=seconds)" "$carrot_count" "$total_count" >> "$WATCH_LOG"
  exit 1
fi
