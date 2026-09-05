#!/usr/bin/env bash
set -u

cd /home/leju-suzhou/zjt_ws/token-cd

PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ART=artifacts/vanilla_recon_shr_canonical_0_299_v2
PP=/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
TASK=widowx_stack_cube
LO=80
HI=99
GPU=4
JOB=44
LOCK="$ART/logs/st_shr_beta100_0_99_job_${JOB}.lock"
LOG="$ART/logs/st_shr_beta100_0_99_job_${JOB}_${TASK}_${LO}_${HI}.log"

completed_in_shard() {
  local seed count=0 dir="$ART/episodes/$TASK/st_shr_beta100"
  for seed in $(seq "$LO" "$HI"); do
    [ -f "$dir/episode_$(printf '%03d' "$seed")_summary.json" ] && \
      [ -f "$dir/episode_$(printf '%03d' "$seed")_arrays.npz" ] && \
      count=$((count + 1))
  done
  printf '%s' "$count"
}

(
  flock -n 9 || exit 0
  while :; do
    count=$(completed_in_shard)
    if [ "$count" -eq 20 ]; then
      touch "$LOG.done"
      exit 0
    fi
    printf '%s extra-gpu4 complete=%s/20\n' "$(date --iso-8601=seconds)" "$count" >> "$LOG"
    env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PP" \
      "$PY" research/semantic_token_cd/st_shr_canonical_rollout.py \
        --artifact "$ART" --snapshot-artifact "$ART" \
        --task "$TASK" --beta 1.0 --seeds "$LO-$HI" --gpu "$GPU" \
        --worker-id st-shr-extra-gpu4-stack80-99 \
        >> "$LOG" 2>&1
    status=$?
    count=$(completed_in_shard)
    [ "$count" -eq 20 ] && continue
    printf '%s retry status=%s complete=%s/20\n' \
      "$(date --iso-8601=seconds)" "$status" "$count" >> "$LOG"
    sleep 3
  done
) 9> "$LOCK"
