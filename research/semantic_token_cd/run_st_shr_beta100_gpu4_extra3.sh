#!/usr/bin/env bash
set -u

cd /home/leju-suzhou/zjt_ws/token-cd

PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ART=artifacts/vanilla_recon_shr_canonical_0_299_v2
PP=/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
TASK=widowx_put_eggplant_in_basket
GPU=4
JOBS=("32|40|59" "33|60|79" "34|80|99")

completed_in_shard() {
  local lo=$1 hi=$2 seed count=0 dir
  dir="$ART/episodes/$TASK/st_shr_beta100"
  for seed in $(seq "$lo" "$hi"); do
    [ -f "$dir/episode_$(printf '%03d' "$seed")_summary.json" ] && \
      [ -f "$dir/episode_$(printf '%03d' "$seed")_arrays.npz" ] && \
      count=$((count + 1))
  done
  printf '%s' "$count"
}

run_one() {
  local spec=$1 i lo hi lock log count status
  IFS='|' read -r i lo hi <<< "$spec"
  lock="$ART/logs/st_shr_beta100_0_99_job_${i}.lock"
  log="$ART/logs/st_shr_beta100_0_99_job_${i}_${TASK}_${lo}_${hi}.log"
  (
    flock -n 9 || return 0
    while :; do
      count=$(completed_in_shard "$lo" "$hi")
      if [ "$count" -eq 20 ]; then
        touch "$log.done"
        return 0
      fi
      printf '%s extra-gpu4 complete=%s/20\n' "$(date --iso-8601=seconds)" "$count" >> "$log"
      env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PP" \
        "$PY" research/semantic_token_cd/st_shr_canonical_rollout.py \
          --artifact "$ART" --snapshot-artifact "$ART" \
          --task "$TASK" --beta 1.0 --seeds "$lo-$hi" --gpu "$GPU" \
          --worker-id "st-shr-extra-gpu4-job${i}" \
          >> "$log" 2>&1
      status=$?
      count=$(completed_in_shard "$lo" "$hi")
      [ "$count" -eq 20 ] && continue
      printf '%s retry status=%s complete=%s/20\n' \
        "$(date --iso-8601=seconds)" "$status" "$count" >> "$log"
      sleep 3
    done
  ) 9> "$lock"
}

for job in "${JOBS[@]}"; do
  run_one "$job" &
done
wait
