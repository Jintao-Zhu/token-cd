#!/usr/bin/env bash
# Sync this checkout to the dev_zjt bind mount, then run the 3-step smoke test.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DOCKER_REPO="${LIBERO_DOCKER_REPO_MIRROR:-/data/docker/dev_zjt/data/code/token-cd}"
CONTAINER="${LIBERO_DOCKER_CONTAINER:-dev_zjt_container}"

mkdir -p "$DOCKER_REPO"
(cd "$REPO_ROOT" && tar --exclude='./.git' -cf - .) | (cd "$DOCKER_REPO" && tar -xf -)
docker exec "$CONTAINER" bash -lc 'cd /root/code/token-cd && bash scripts/run_l11_matched_libero_smoke.sh'
