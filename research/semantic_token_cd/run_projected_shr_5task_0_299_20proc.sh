#!/usr/bin/env bash
set -u

cd /home/leju-suzhou/zjt_ws/token-cd

PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ART=artifacts/projected_shr_5task_0_299_v1
SOURCE=artifacts/vanilla_recon_shr_canonical_0_299_v2
PP=/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
TASKS=(google_robot_close_drawer google_robot_open_drawer google_robot_move_near google_robot_pick_coke_can widowx_carrot_on_plate)
GPUS=(1 2 3 4 5)
SHARD_SIZE=75
ARMS=(proj_shr_eta000 proj_shr_eta025)

mkdir -p "$ART/logs" "$ART/episodes"

completed_in_shard() {
  local task=$1 lo=$2 hi=$3 arm seed count=0
  for arm in "${ARMS[@]}"; do
    for seed in $(seq "$lo" "$hi"); do
      dir="$ART/episodes/$task/$arm"
      if [ -f "$dir/episode_$(printf '%03d' "$seed")_summary.json" ] && \
         [ -f "$dir/episode_$(printf '%03d' "$seed")_arrays.npz" ]; then
        count=$((count + 1))
      fi
    done
  done
  printf '%s' "$count"
}

run_shard() {
  local task=$1 gpu=$2 worker=$3 lo=$4 hi=$5 expected log lock count status
  expected=$((2 * (hi - lo + 1)))
  log="$ART/logs/${task}_${lo}_${hi}.log"
  lock="$ART/logs/${task}_${lo}_${hi}.lock"
  (
    flock -n 9 || return 0
    while :; do
      count=$(completed_in_shard "$task" "$lo" "$hi")
      if [ "$count" -eq "$expected" ]; then
        touch "$log.done"
        return 0
      fi
      printf '%s gpu=%s complete=%s/%s\n' \
        "$(date --iso-8601=seconds)" "$gpu" "$count" "$expected" >> "$log"
      env \
        HF_HUB_OFFLINE=1 \
        CUDA_VISIBLE_DEVICES="$gpu" \
        TOKENIZERS_PARALLELISM=false \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        OMP_NUM_THREADS=2 \
        MKL_NUM_THREADS=2 \
        PYTHONPATH="$PP" \
        "$PY" research/semantic_token_cd/projected_shr_rollout.py \
          --artifact "$ART" \
          --snapshot-artifact "$SOURCE" \
          --task "$task" \
          --seeds "$lo-$hi" \
          --gpu "$gpu" \
          --worker-id "$worker" >> "$log" 2>&1
      status=$?
      count=$(completed_in_shard "$task" "$lo" "$hi")
      [ "$count" -eq "$expected" ] && continue
      printf '%s retry status=%s complete=%s/%s\n' \
        "$(date --iso-8601=seconds)" "$status" "$count" "$expected" >> "$log"
      sleep 3
    done
  ) 9> "$lock"
}

for task_index in 0 1 2 3 4; do
  task=${TASKS[$task_index]}
  gpu=${GPUS[$task_index]}
  for worker_index in 0 1 2 3; do
    lo=$((worker_index * SHARD_SIZE))
    hi=$((lo + SHARD_SIZE - 1))
    run_shard "$task" "$gpu" \
      "projected-shr-gpu${gpu}-worker${worker_index}" "$lo" "$hi" &
  done
done
wait

count=$(find "$ART/episodes" -type f -name 'episode_*_summary.json' | wc -l)
if [ "$count" -eq 3000 ]; then
  printf 'Projected-SHR experiment complete: %s/3000\n' "$count" > "$ART/logs/COMPLETE"
fi
