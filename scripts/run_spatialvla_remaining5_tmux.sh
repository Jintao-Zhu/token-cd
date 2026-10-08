#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD="$ROOT/token-cd"
PYTHON="$ROOT/openvla-ar-h100/bin/python"
RUNNER="$TOKEN_CD/research/semantic_token_cd/spatialvla_l11_matched_remaining_rollout.py"
ARTIFACT=artifacts/spatialvla_simpler_l11matched_remaining5_n100
export PYTHONPATH="$ROOT/spatialvla-runtime-overlay-20260928:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/pcd_openvla_simpler_box_31b027e/source:$TOKEN_CD/task1/shim_site:$TOKEN_CD"

gpu=${1:?usage: run_spatialvla_remaining5_tmux.sh GPU_ID}
case "$gpu" in
  2) seeds=100-149 ;;
  3) seeds=150-199 ;;
  *) echo "Only GPUs 2 and 3 are assigned to this run" >&2; exit 2 ;;
esac

cd "$TOKEN_CD"
mkdir -p logs
echo "[$(date -Is)] GPU $gpu begins Fractal / place-apple shard seeds $seeds"
"$PYTHON" -u "$RUNNER" \
  --artifact "$ARTIFACT" --gpu "$gpu" --worker-id "gpu${gpu}_fractal" \
  --model-setup google_robot \
  --tasks google_robot_place_apple_in_closed_top_drawer --seeds "$seeds" \
  >"logs/spatialvla_remaining_gpu${gpu}_fractal.log" 2>&1

echo "[$(date -Is)] GPU $gpu begins Bridge / four-WidowX shard seeds $seeds"
"$PYTHON" -u "$RUNNER" \
  --artifact "$ARTIFACT" --gpu "$gpu" --worker-id "gpu${gpu}_bridge" \
  --model-setup widowx_bridge \
  --tasks widowx_carrot_on_plate,widowx_put_eggplant_in_basket,widowx_spoon_on_towel,widowx_stack_cube \
  --seeds "$seeds" \
  >"logs/spatialvla_remaining_gpu${gpu}_bridge.log" 2>&1

echo "[$(date -Is)] GPU $gpu finished its SpatialVLA remaining-task shard"
