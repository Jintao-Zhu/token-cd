#!/usr/bin/env bash
set -uo pipefail

cd /home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
SOURCE=artifacts/vanilla_recon_shr_canonical_0_299_v2
ART=artifacts/prompt_attn_shr_v1
PP=/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
mkdir -p "$ART/rollout_logs"

run_job() {
  local task=$1 seeds=$2 gpu=$3 name status attempt
  name="recovery_${task}_${seeds//-/_}_gpu${gpu}"
  for attempt in 1 2 3; do
    HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PP" \
      "$PY" research/semantic_token_cd/prompt_attn_shr_rollout.py \
      --artifact "$ART" --snapshot-artifact "$SOURCE" --task "$task" \
      --seeds "$seeds" --gpu "$gpu" --worker-id "$name" \
      >"$ART/rollout_logs/${name}_attempt${attempt}.log" 2>&1
    status=$?
    if [ "$status" -eq 0 ]; then
      return 0
    fi
    sleep 10
  done
  return "$status"
}

# GPUs 1/4/5 reject a second concurrent Vulkan renderer.  Healthy GPUs 0/2/3
# have enough measured headroom for one additional model (four total), so each
# recovery lane processes two missing shards sequentially.
{
  run_job google_robot_pick_coke_can 17-33 0
  run_job google_robot_move_near 17-33 0
} &
{
  run_job google_robot_pick_coke_can 68-83 2
  run_job google_robot_move_near 68-83 2
} &
{
  run_job google_robot_pick_coke_can 84-99 3
  run_job google_robot_move_near 84-99 3
} &
wait

# Whichever coordinator finishes last owns final analysis.
while true; do
  summaries=$(find "$ART/episodes" -type f -name 'episode_*_summary.json' | wc -l)
  if [ "$summaries" -eq 900 ]; then
    HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PP" \
      "$PY" research/semantic_token_cd/analyze_prompt_attn_shr.py --artifact "$ART" \
      >"$ART/rollout_logs/final_analysis.log" 2>&1
    printf '900/900 Prompt-Attn-SHR v1 arm-episodes complete\n' >"$ART/rollout_logs/COMPLETE"
    exit 0
  fi
  sleep 30
done
