#!/usr/bin/env bash
set -u

cd /home/leju-suzhou/zjt_ws/token-cd

PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ART=artifacts/vanilla_recon_shr_canonical_0_299_v2
PP=/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source

# These shards previously failed and their original fixed slots have exited.
# Active/future shards still owned by the 3-GPU launcher are intentionally absent.
JOBS=(
  "89|widowx_carrot_on_plate|280|299|4"
  "93|widowx_put_eggplant_in_basket|60|79|4"
  "94|widowx_put_eggplant_in_basket|80|99|4"
  "95|widowx_put_eggplant_in_basket|100|119|1"
  "97|widowx_put_eggplant_in_basket|140|159|5"
)

run_one() {
  local spec=$1 i task lo hi gpu log claim
  IFS='|' read -r i task lo hi gpu <<< "$spec"
  log="$ART/logs/job_${i}_${task}_${lo}_${hi}.log"
  claim="$ART/logs/job_${i}.tail_accelerate.claim"

  [ -f "$log.done" ] && return 0
  if ! mkdir "$claim" 2>/dev/null; then
    printf 'skip job %s: already claimed\n' "$i"
    return 0
  fi

  printf '\n--- tail acceleration retry on GPU %s ---\n' "$gpu" >> "$log"
  env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PP" \
    "$PY" research/semantic_token_cd/semantic_recon_rollout.py \
      --artifact "$ART" --task "$task" --gpu "$gpu" --seeds "$lo-$hi" \
      >> "$log" 2>&1
  status=$?
  if [ "$status" -eq 0 ]; then
    touch "$log.done"
  fi
  rmdir "$claim"
  return "$status"
}

for job in "${JOBS[@]}"; do
  run_one "$job" &
done
wait

