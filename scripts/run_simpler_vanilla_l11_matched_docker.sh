#!/usr/bin/env bash
# Run paired SIMPLER vanilla and L11-Matched episodes in dev_zjt_container.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DOCKER_REPO="${SIMPLER_DOCKER_REPO_MIRROR:-/data/docker/dev_zjt/data/code/token-cd}"
CONTAINER="${LIBERO_DOCKER_CONTAINER:-dev_zjt_container}"

"$SCRIPT_DIR/prepare_simpler_docker.sh"
mkdir -p "$DOCKER_REPO"
(cd "$REPO_ROOT" && tar --exclude='./.git' --exclude='*/__pycache__' --exclude='*.pyc' -cf - .) | \
  (cd "$DOCKER_REPO" && tar -xf -)

docker exec "$CONTAINER" bash -lc '
  set -euo pipefail
  cd /root/code/token-cd
  source scripts/activate_simpler_docker.sh
  exec "$SIMPLER_ENV/bin/python" research/semantic_token_cd/simpler_vanilla_l11_matched_rollout.py \
    --artifact "${SIMPLER_ARTIFACT:-$TOKEN_CD/artifacts/simpler_vanilla_l11_matched_v1}" \
    --gpu 0 "$@"
' bash "$@"
