#!/usr/bin/env bash
# One 3-step LIBERO-Spatial smoke episode; run inside dev_zjt_container.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/activate_libero_docker.sh"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1

TASK="pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
ARTIFACT="${L11_SMOKE_ARTIFACT:-$TOKEN_CD/artifacts/l11_matched_libero_smoke_v1}"

"$LIBERO_ENV/bin/python" research/semantic_token_cd/libero_matched_rollout.py \
  --artifact "$ARTIFACT" \
  --checkpoint "$OPENVLA_CHECKPOINT" \
  --dataset-statistics "$OPENVLA_CHECKPOINT/dataset_statistics.json" \
  --task "$TASK" \
  --episodes 0 \
  --suite libero_spatial \
  --unnorm-key libero_spatial \
  --gpu 0 \
  --entity-mode source_target \
  --query-mode instruction_only \
  --attention-layers 11 \
  --env-seed 0 \
  --settle-steps 1 \
  --max-steps 3 \
  --smoke
