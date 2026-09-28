#!/usr/bin/env bash
# Sync this checkout and run a full L11-Matched LIBERO-Spatial rollout in Docker.
# Pass normal libero_matched_rollout.py arguments, e.g. --task TASK --episodes 0-9.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DOCKER_REPO="${LIBERO_DOCKER_REPO_MIRROR:-/data/docker/dev_zjt/data/code/token-cd}"
CONTAINER="${LIBERO_DOCKER_CONTAINER:-dev_zjt_container}"
L11_RUN_ARTIFACT="${L11_RUN_ARTIFACT:-/root/code/token-cd/artifacts/libero_l11_matched_docker_v1}"

mkdir -p "$DOCKER_REPO"
(cd "$REPO_ROOT" && tar --exclude='./.git' -cf - .) | (cd "$DOCKER_REPO" && tar -xf -)

docker exec -e "L11_RUN_ARTIFACT=$L11_RUN_ARTIFACT" "$CONTAINER" bash -lc '
  set -euo pipefail
  cd /root/code/token-cd
  source scripts/activate_libero_docker.sh
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export HF_HUB_OFFLINE=1
  exec "$LIBERO_ENV/bin/python" research/semantic_token_cd/libero_matched_rollout.py \
    --artifact "${L11_RUN_ARTIFACT:-$TOKEN_CD/artifacts/libero_l11_matched_docker_v1}" \
    --checkpoint "$OPENVLA_CHECKPOINT" \
    --dataset-statistics "$OPENVLA_CHECKPOINT/dataset_statistics.json" \
    --suite libero_spatial --unnorm-key libero_spatial --gpu 0 \
    --entity-mode source_target --query-mode instruction_only --attention-layers 11 \
    "$@"
' bash "$@"
