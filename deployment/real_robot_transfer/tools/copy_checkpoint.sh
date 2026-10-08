#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 SOURCE_MODEL_DIR user@host:/destination/model-dir" >&2
  exit 2
fi

source_dir="${1%/}"
destination="${2%/}/"
if [[ ! -f "$source_dir/config.json" || ! -f "$source_dir/model.safetensors.index.json" ]]; then
  echo "Not an OpenVLA sharded checkpoint directory: $source_dir" >&2
  exit 2
fi

rsync -a --partial --append-verify --info=progress2 "$source_dir/" "$destination"
