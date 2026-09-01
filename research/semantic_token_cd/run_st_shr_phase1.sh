#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../.."

PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
SOURCE=/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
ART=artifacts/st_shr_cd_v1
STATE="$ART/queue"
LOGS="$ART/logs"
GPUS=(1 2 3 4 5 6 7)
WORKERS_PER_GPU=3
TASKS=(
  google_robot_pick_coke_can google_robot_open_drawer google_robot_close_drawer
  google_robot_move_near google_robot_place_apple_in_closed_top_drawer
  widowx_carrot_on_plate widowx_put_eggplant_in_basket
  widowx_spoon_on_towel widowx_stack_cube
)
BETAS=(0.25 1.0 4.0)

mkdir -p "$STATE" "$LOGS"
QUEUE="$STATE/jobs.tsv"
COUNTER="$STATE/counter"
LOCK="$STATE/queue.lock"

if [[ ! -f "$QUEUE" ]]; then
  tmp="$QUEUE.tmp.$$"
  : > "$tmp"
  for lo in $(seq 0 10 90); do
    hi=$((lo + 9))
    for task in "${TASKS[@]}"; do
      for beta in "${BETAS[@]}"; do
        printf '%s\t%s\t%d-%d\n' "$task" "$beta" "$lo" "$hi" >> "$tmp"
      done
    done
  done
  mv "$tmp" "$QUEUE"
fi
[[ -f "$COUNTER" ]] || printf '0\n' > "$COUNTER"

worker() {
  local gpu=$1 worker_id=$2
  while true; do
    local index line task beta seeds job rc=1
    exec 9>"$LOCK"
    flock 9
    index=$(<"$COUNTER")
    line=$(sed -n "$((index + 1))p" "$QUEUE")
    if [[ -z "$line" ]]; then
      flock -u 9
      return 0
    fi
    printf '%d\n' "$((index + 1))" > "$COUNTER"
    flock -u 9
    IFS=$'\t' read -r task beta seeds <<< "$line"
    job=$(printf '%03d_%s_b%s_s%s' "$index" "$task" "${beta//./}" "${seeds/-/_}")
    for attempt in 1 2 3; do
      echo "[$(date --iso-8601=seconds)] START gpu=$gpu worker=$worker_id attempt=$attempt $line" >> "$LOGS/orchestrator.log"
      HF_HUB_OFFLINE=1 PYTHONPATH="$PWD:$SOURCE" "$PY" \
        research/semantic_token_cd/st_shr_rollout.py \
        --artifact "$ART" --task "$task" --beta "$beta" --seeds "$seeds" \
        --gpu "$gpu" --worker-id "$worker_id" > "$LOGS/$job.log" 2>&1
      rc=$?
      if [[ $rc -eq 0 ]]; then
        touch "$LOGS/$job.done"
        echo "[$(date --iso-8601=seconds)] DONE gpu=$gpu worker=$worker_id $line" >> "$LOGS/orchestrator.log"
        break
      fi
      echo "[$(date --iso-8601=seconds)] RETRY rc=$rc gpu=$gpu worker=$worker_id $line" >> "$LOGS/orchestrator.log"
    done
    if [[ $rc -ne 0 ]]; then
      touch "$LOGS/$job.failed"
      echo "[$(date --iso-8601=seconds)] FAILED gpu=$gpu worker=$worker_id $line" >> "$LOGS/orchestrator.log"
    fi
  done
}

echo "[$(date --iso-8601=seconds)] LAUNCH jobs=$(wc -l < "$QUEUE") GPUs=${GPUS[*]} workers_per_gpu=$WORKERS_PER_GPU" >> "$LOGS/orchestrator.log"
pids=()
for gpu in "${GPUS[@]}"; do
  for slot in $(seq 0 $((WORKERS_PER_GPU - 1))); do
    worker "$gpu" "gpu${gpu}_slot${slot}" &
    pids+=("$!")
  done
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
echo "[$(date --iso-8601=seconds)] COMPLETE rc=$rc summaries=$(find "$ART/episodes" -name 'episode_*_summary.json' 2>/dev/null | wc -l)" >> "$LOGS/orchestrator.log"
exit "$rc"
