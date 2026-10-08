#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD="$ROOT/token-cd"
PYTHON="$ROOT/openvla-ar-h100/bin/python"
RUNNER="$TOKEN_CD/research/semantic_token_cd/spatialvla_three_studies_rollout.py"
ARTIFACT="$TOKEN_CD/artifacts/spatialvla_l10_vanilla_rtfull7_to300_v1"
SNAPSHOT_ARTIFACT="$TOKEN_CD/artifacts/vanilla_recon_shr_canonical_0_299_v2"
REFERENCE_ARTIFACT="$TOKEN_CD/artifacts/simpler_rt_full_7task_4200_v1"
STUDY=remaining7_rtfull_expand300
RESUME_TAG=resume_20261001

export PYTHONPATH="$ROOT/spatialvla-runtime-overlay-20260928:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/pcd_openvla_simpler_box_31b027e/source:$TOKEN_CD/task1/shim_site:$TOKEN_CD"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

cd "$TOKEN_CD"
mkdir -p "$ARTIFACT/logs"

[[ -f "$ARTIFACT/CONFIG_LOCK.json" ]] || { echo "missing frozen config lock: $ARTIFACT" >&2; exit 2; }
[[ -d "$SNAPSHOT_ARTIFACT/snapshots" ]] || { echo "missing canonical snapshots: $SNAPSHOT_ARTIFACT" >&2; exit 2; }
[[ -d "$REFERENCE_ARTIFACT/episodes" ]] || { echo "missing paired RT-full reference: $REFERENCE_ARTIFACT" >&2; exit 2; }

# Preserve the original, disjoint twelve-worker assignment. The runner skips
# an arm/seed as soon as its complete summary exists, so this resumes only
# missing outputs without changing any completed record.
workers=(
  '1 google_robot_pick_coke_can,google_robot_move_near 0-49,200-249 google_robot'
  '1 google_robot_place_apple_in_closed_top_drawer 0-49,200-249 google_robot'
  '1 widowx_spoon_on_towel 0-49,200-249 widowx_bridge'
  '1 widowx_stack_cube 0-49,200-249 widowx_bridge'
  '2 google_robot_pick_coke_can 50-99,250-299 google_robot'
  '2 google_robot_move_near 50-99,250-299 google_robot'
  '2 widowx_carrot_on_plate,widowx_put_eggplant_in_basket 0-49,200-249 widowx_bridge'
  '2 widowx_spoon_on_towel 50-99,250-299 widowx_bridge'
  '3 google_robot_place_apple_in_closed_top_drawer 50-99,250-299 google_robot'
  '3 widowx_carrot_on_plate 50-99,250-299 widowx_bridge'
  '3 widowx_put_eggplant_in_basket 50-99,250-299 widowx_bridge'
  '3 widowx_stack_cube 50-99,250-299 widowx_bridge'
)

for row in "${workers[@]}"; do
  read -r gpu tasks seeds setup <<<"$row"
  session="svla_l10_rtfull300_${RESUME_TAG}_g${gpu}_${tasks%%,*}_${seeds//[,-]/_}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 2
  fi
done

for row in "${workers[@]}"; do
  read -r gpu tasks seeds setup <<<"$row"
  session="svla_l10_rtfull300_${RESUME_TAG}_g${gpu}_${tasks%%,*}_${seeds//[,-]/_}"
  log="$ARTIFACT/logs/${session}.log"
  tmux new-session -d -s "$session" \
    "cd '$TOKEN_CD' && export PYTHONPATH='$PYTHONPATH' HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json && '$PYTHON' -u '$RUNNER' --study '$STUDY' --artifact '$ARTIFACT' --snapshot-artifact '$SNAPSHOT_ARTIFACT' --reference-artifact '$REFERENCE_ARTIFACT' --gpu '$gpu' --worker-id '$session' --model-setup '$setup' --tasks '$tasks' --seeds '$seeds' > '$log' 2>&1"
  echo "started tmux=$session tasks=$tasks seeds=$seeds GPU=$gpu setup=$setup log=$log"
done

echo "resume protocol=SPATIALVLA_L10_VANILLA_REMAINING7_RTFULL_SEEDS0_299_V1 workers=12 GPUs=1,2,3; prior summaries are skipped"
