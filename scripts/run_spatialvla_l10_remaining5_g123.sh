#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD="$ROOT/token-cd"
PYTHON="$ROOT/openvla-ar-h100/bin/python"
RUNNER="$TOKEN_CD/research/semantic_token_cd/spatialvla_l10_matched_remaining_rollout.py"
SOURCE="$TOKEN_CD/artifacts/spatialvla_simpler_l11matched_remaining5_n100"
ARTIFACT="$TOKEN_CD/artifacts/spatialvla_l10_matched_remaining5_n100"
export PYTHONPATH="$ROOT/spatialvla-runtime-overlay-20260928:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/pcd_openvla_simpler_box_31b027e/source:$TOKEN_CD/task1/shim_site:$TOKEN_CD"

cd "$TOKEN_CD"
mkdir -p "$ARTIFACT/logs"

# Keep four total SpatialVLA workers per card: GPU1 has four new workers;
# GPUs2/3 currently each have two L10-vs-L11 drawer workers, so add two each.
workers=(
  '1|google_robot_place_apple_in_closed_top_drawer|google_robot|100-199'
  '1|widowx_carrot_on_plate|widowx_bridge|100-149'
  '1|widowx_put_eggplant_in_basket|widowx_bridge|100-149'
  '1|widowx_spoon_on_towel|widowx_bridge|100-149'
  '2|widowx_carrot_on_plate|widowx_bridge|150-199'
  '2|widowx_put_eggplant_in_basket|widowx_bridge|150-199'
  '3|widowx_spoon_on_towel|widowx_bridge|150-199'
  '3|widowx_stack_cube|widowx_bridge|100-199'
)

for item in "${workers[@]}"; do
  IFS='|' read -r gpu task setup seeds <<< "$item"
  session="spatialvla_l10_remaining_g${gpu}_${task}_${seeds}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 2
  fi
done

for item in "${workers[@]}"; do
  IFS='|' read -r gpu task setup seeds <<< "$item"
  session="spatialvla_l10_remaining_g${gpu}_${task}_${seeds}"
  log="$ARTIFACT/logs/${task}_gpu${gpu}_seeds_${seeds}.log"
  tmux new-session -d -s "$session" \
    "cd '$TOKEN_CD' && export PYTHONPATH='$PYTHONPATH' HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json && '$PYTHON' -u '$RUNNER' --artifact '$ARTIFACT' --snapshot-artifact '$SOURCE' --gpu '$gpu' --worker-id 'gpu${gpu}_${task}_${seeds}' --model-setup '$setup' --tasks '$task' --seeds '$seeds' > '$log' 2>&1"
  echo "started tmux=$session task=$task setup=$setup GPU=$gpu seeds=$seeds log=$log"
done

echo "protocol=SPATIALVLA_SIMPLER_L10_MATCHED_REMAINING5_V1 total_episodes=500 new_L10_arm=500 seeds=100-199 GPUs=1,2,3 workers=8 (four total per GPU counting two existing workers on GPUs2/3)"
