#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 create MODEL_DIR [MANIFEST_PATH]" >&2
  echo "       $0 verify MODEL_DIR MANIFEST_PATH" >&2
}

if [[ $# -lt 2 ]]; then usage; exit 2; fi
mode="$1"
model_dir="${2%/}"

case "$mode" in
  create)
    manifest="${3:-$model_dir/SHA256SUMS}"
    if [[ ! -d "$model_dir" ]]; then echo "Missing model directory: $model_dir" >&2; exit 2; fi
    (
      cd "$model_dir"
      find . -type f ! -path './.cache/*' ! -name 'SHA256SUMS' -print0 \
        | sort -z \
        | xargs -0 sha256sum
    ) > "$manifest"
    echo "Wrote $manifest"
    ;;
  verify)
    if [[ $# -ne 3 ]]; then usage; exit 2; fi
    manifest="$3"
    if [[ ! -f "$manifest" ]]; then echo "Missing manifest: $manifest" >&2; exit 2; fi
    (cd "$model_dir" && sha256sum --check "$manifest")
    ;;
  *) usage; exit 2 ;;
esac
