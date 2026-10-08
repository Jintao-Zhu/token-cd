#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD="$ROOT/token-cd"
PYTHON="$ROOT/openvla-ar-h100/bin/python"
RUNNER="$TOKEN_CD/research/semantic_token_cd/spatialvla_l10_matched_rtfull7_rollout.py"
ARTIFACT="$TOKEN_CD/artifacts/spatialvla_l10_matched_rtfull7_n100_199"
SNAPSHOT_ARTIFACT="$TOKEN_CD/artifacts/vanilla_recon_shr_canonical_0_299_v2"
REFERENCE_ARTIFACT="$TOKEN_CD/artifacts/simpler_rt_full_7task_4200_v1"
export PYTHONPATH="$ROOT/spatialvla-runtime-overlay-20260928:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/pcd_openvla_simpler_box_31b027e/source:$TOKEN_CD/task1/shim_site:$TOKEN_CD"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

cd "$TOKEN_CD"
mkdir -p "$ARTIFACT/logs"

# Each row is one independent process / tmux session. Together these shards
# cover 7 tasks x seeds 100-199 exactly once: 700 L10-Matched episodes.
workers=(
  '1 google_robot_pick_coke_can 100-199 google_robot'
  '1 widowx_carrot_on_plate 100-149 widowx_bridge'
  '1 widowx_put_eggplant_in_basket 100-149 widowx_bridge'
  '1 widowx_spoon_on_towel 100-149 widowx_bridge'
  '2 google_robot_move_near 100-199 google_robot'
  '2 widowx_carrot_on_plate 150-174 widowx_bridge'
  '2 widowx_put_eggplant_in_basket 150-199 widowx_bridge'
  '2 widowx_stack_cube 100-149 widowx_bridge'
  '3 google_robot_place_apple_in_closed_top_drawer 100-199 google_robot'
  '3 widowx_carrot_on_plate 175-199 widowx_bridge'
  '3 widowx_spoon_on_towel 150-199 widowx_bridge'
  '3 widowx_stack_cube 150-199 widowx_bridge'
)

# Refuse partial/duplicate reruns under this new protocol; this run is fresh.
if [[ -e "$ARTIFACT/CONFIG_LOCK.json" ]]; then
  echo "artifact already initialized; refusing to overwrite/repartition: $ARTIFACT" >&2
  exit 2
fi
for gpu in 1 2 3; do
  n=0
  for row in "${workers[@]}"; do
    read -r g _ <<<"$row"
    if [[ "$g" == "$gpu" ]]; then n=$((n + 1)); fi
  done
  [[ "$n" == 4 ]] || { echo "expected 4 workers on GPU $gpu, got $n" >&2; exit 2; }
done
for row in "${workers[@]}"; do
  read -r gpu task seeds setup <<<"$row"
  session="svla_l10_rtfull_g${gpu}_${task}_${seeds//-/_}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 2
  fi
done

for row in "${workers[@]}"; do
  read -r gpu task seeds setup <<<"$row"
  session="svla_l10_rtfull_g${gpu}_${task}_${seeds//-/_}"
  log="$ARTIFACT/logs/${session}.log"
  tmux new-session -d -s "$session" \
    "cd '$TOKEN_CD' && export PYTHONPATH='$PYTHONPATH' HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json && '$PYTHON' -u '$RUNNER' --artifact '$ARTIFACT' --snapshot-artifact '$SNAPSHOT_ARTIFACT' --reference-artifact '$REFERENCE_ARTIFACT' --gpu '$gpu' --worker-id '$session' --model-setup '$setup' --tasks '$task' --seeds '$seeds' > '$log' 2>&1"
  echo "started tmux=$session task=$task seeds=$seeds GPU=$gpu checkpoint=$setup log=$log"
done

echo "protocol=SPATIALVLA_SIMPLER_L10_MATCHED_RTFULL_7TASK_100_199_V1 total=700 workers=12 (4 each on GPUs 1,2,3)"
