#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD="$ROOT/token-cd"
PYTHON="$ROOT/openvla-ar-h100/bin/python"
RUNNER="$TOKEN_CD/research/semantic_token_cd/spatialvla_l11_matched_remaining_rollout.py"
ARTIFACT=artifacts/spatialvla_simpler_l11matched_remaining5_n100

gpu=${1:?usage: run_spatialvla_remaining5_worker.sh GPU_ID ROLE}
role=${2:?usage: run_spatialvla_remaining5_worker.sh GPU_ID ROLE}
case "$gpu" in
  2) seeds=100-149 ;;
  3) seeds=150-199 ;;
  *) echo "Only GPUs 2 and 3 are assigned to this run" >&2; exit 2 ;;
esac
case "$role" in
  fractal)
    setup=google_robot
    tasks=google_robot_place_apple_in_closed_top_drawer
    ;;
  bridge_a)
    setup=widowx_bridge
    tasks=widowx_carrot_on_plate,widowx_put_eggplant_in_basket
    ;;
  bridge_b)
    setup=widowx_bridge
    tasks=widowx_spoon_on_towel
    ;;
  bridge_c)
    setup=widowx_bridge
    tasks=widowx_stack_cube
    ;;
  *) echo "Unknown worker role: $role" >&2; exit 2 ;;
esac

export PYTHONPATH="$ROOT/spatialvla-runtime-overlay-20260928:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/pcd_openvla_simpler_box_31b027e/source:$TOKEN_CD/task1/shim_site:$TOKEN_CD"
cd "$TOKEN_CD"
mkdir -p logs
worker_id="gpu${gpu}_${role}"
"$PYTHON" -u "$RUNNER" \
  --artifact "$ARTIFACT" --gpu "$gpu" --worker-id "$worker_id" \
  --model-setup "$setup" --tasks "$tasks" --seeds "$seeds" \
  >"logs/spatialvla_remaining_${worker_id}.log" 2>&1
