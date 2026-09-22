#!/usr/bin/env bash
set -euo pipefail
REPO="$1"; DEST="$2"; LOG="$3"
mkdir -p "$(dirname "$LOG")"
/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python - "$REPO" "$DEST" > "$LOG" 2>&1 <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
print(f'downloading {repo} -> {dest}', flush=True)
snapshot_download(repo_id=repo, local_dir=dest, max_workers=4)
print('DONE', flush=True)
PY
