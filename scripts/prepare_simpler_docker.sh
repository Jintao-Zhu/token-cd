#!/usr/bin/env bash
# Install the Vulkan loader in the running container and expose its NVIDIA ICD.
set -euo pipefail

CONTAINER="${LIBERO_DOCKER_CONTAINER:-dev_zjt_container}"
HOST_NVIDIA_ICD="${HOST_NVIDIA_ICD:-/usr/share/vulkan/icd.d/nvidia_icd.json}"

if [[ ! -r "$HOST_NVIDIA_ICD" ]]; then
  echo "NVIDIA Vulkan ICD metadata not found: $HOST_NVIDIA_ICD" >&2
  exit 1
fi

docker exec "$CONTAINER" bash -lc '
  if ! ldconfig -p 2>/dev/null | grep -q "libvulkan.so.1" || ! command -v tmux >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y libvulkan1 tmux
  fi
  mkdir -p /usr/share/vulkan/icd.d
'
docker cp "$HOST_NVIDIA_ICD" "$CONTAINER:/usr/share/vulkan/icd.d/nvidia_icd.json"
docker exec "$CONTAINER" test -r /usr/share/vulkan/icd.d/nvidia_icd.json
