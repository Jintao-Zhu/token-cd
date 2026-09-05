#!/usr/bin/env bash
set -u

cd /home/leju-suzhou/zjt_ws/token-cd

PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ART=artifacts/projected_shr_5task_0_299_v1
SOURCE=artifacts/vanilla_recon_shr_canonical_0_299_v2
PP=/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
TASKS=(google_robot_close_drawer google_robot_open_drawer google_robot_move_near google_robot_pick_coke_can widowx_carrot_on_plate)
WORKERS=(4 4 2 1 1)
GPUS=(1 4 5)
ARMS=(proj_shr_eta000 proj_shr_eta025)

mkdir -p "$ART/logs" "$ART/episodes"

seed_spec() {
  local worker=$1 nworkers=$2 seed spec=""
  for ((seed=worker; seed<300; seed+=nworkers)); do
    if [ -n "$spec" ]; then spec+=","; fi
    spec+="$seed"
  done
  printf '%s' "$spec"
}

completed_in_spec() {
  local task=$1 spec=$2 arm seed dir count=0
  local old_ifs=$IFS
  IFS=,
  read -ra seeds <<< "$spec"
  IFS=$old_ifs
  for arm in "${ARMS[@]}"; do
    for seed in "${seeds[@]}"; do
      dir="$ART/episodes/$task/$arm"
      if [ -f "$dir/episode_$(printf '%03d' "$seed")_summary.json" ] && \
         [ -f "$dir/episode_$(printf '%03d' "$seed")_arrays.npz" ]; then
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
  log="$ART/logs/resume_${task}_${worker}of${nworkers}.log"
  lock="$ART/logs/resume_${task}_${worker}of${nworkers}.lock"
  (
    flock -n 9 || return 0
    while :; do
      count=$(completed_in_spec "$task" "$spec")
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
          --seeds "$spec" \
          --gpu "$gpu" \
          --worker-id "balanced-${task}-${worker}of${nworkers}" >> "$log" 2>&1
      status=$?
      count=$(completed_in_spec "$task" "$spec")
      [ "$count" -eq "$expected" ] && continue
      printf '%s retry status=%s complete=%s/%s\n' \
        "$(date --iso-8601=seconds)" "$status" "$count" "$expected" >> "$log"
      sleep 3
    done
  ) 9> "$lock"
}

job=0
for task_index in 0 1 2 3 4; do
  task=${TASKS[$task_index]}
  nworkers=${WORKERS[$task_index]}
  for ((worker=0; worker<nworkers; worker++)); do
    gpu=${GPUS[$((job % ${#GPUS[@]}))]}
    run_worker "$task" "$nworkers" "$worker" "$gpu" &
    job=$((job + 1))
  done
done
wait

count=$(find "$ART/episodes" -type f -name 'episode_*_summary.json' | wc -l)
if [ "$count" -eq 3000 ]; then
  printf 'Projected-SHR experiment complete: %s/3000\n' "$count" > "$ART/logs/COMPLETE"
fi
