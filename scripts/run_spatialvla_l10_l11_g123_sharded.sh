#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD="$ROOT/token-cd"
PYTHON="$ROOT/openvla-ar-h100/bin/python"
RUNNER="$TOKEN_CD/research/semantic_token_cd/spatialvla_layer_compare_rollout.py"
SOURCE="$TOKEN_CD/artifacts/spatialvla_simpler_l11matched_4task_n100"
ARTIFACT="$TOKEN_CD/artifacts/spatialvla_l10_vs_l11_4task_n100"
export PYTHONPATH="$ROOT/spatialvla-runtime-overlay-20260928:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/pcd_openvla_simpler_box_31b027e/source:$TOKEN_CD/task1/shim_site:$TOKEN_CD"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

cd "$TOKEN_CD"
mkdir -p "$ARTIFACT/logs"
tasks=(
  google_robot_open_drawer
  google_robot_close_drawer
  google_robot_pick_coke_can
  google_robot_move_near
)

for gpu in 1 2 3; do
  case "$gpu" in
    1) seeds=100-133 ;;
    2) seeds=134-166 ;;
    3) seeds=167-199 ;;
  esac
  for task in "${tasks[@]}"; do
    session="spatialvla_l10_l11_g${gpu}_${task}"
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "tmux session already exists: $session" >&2
      exit 2
    fi
  done
done

for gpu in 1 2 3; do
  case "$gpu" in
    1) seeds=100-133 ;;
    2) seeds=134-166 ;;
    3) seeds=167-199 ;;
  esac
  for task in "${tasks[@]}"; do
    session="spatialvla_l10_l11_g${gpu}_${task}"
    log="$ARTIFACT/logs/${task}_gpu${gpu}_seeds_${seeds}.log"
    tmux new-session -d -s "$session" \
      "cd '$TOKEN_CD' && export PYTHONPATH='$PYTHONPATH' HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json && '$PYTHON' -u '$RUNNER' --artifact '$ARTIFACT' --snapshot-artifact '$SOURCE' --gpu '$gpu' --worker-id 'gpu${gpu}_${task}_${seeds}' --tasks '$task' --seeds '$seeds' > '$log' 2>&1"
    echo "started tmux=$session task=$task GPU=$gpu seeds=$seeds log=$log"
  done
done

echo "protocol=SPATIALVLA_SIMPLER_LAYER10_VS_LAYER11_MATCHED_V1 tasks=4 seed-pairs=400 arms=2 total_arm_episodes=800 workers=12 (4/GPU across GPUs 1,2,3)"
