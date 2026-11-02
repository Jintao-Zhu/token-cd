#!/usr/bin/env bash
# Robust per-file download of a pi0-FAST checkpoint from hf-mirror.
#
# The huggingface_hub snapshot download aborts the whole batch on "Max retries exceeded"
# against the AWS CDN that hf-mirror redirects to, and the clash proxy on this host is
# non-functional (subscription refresh fails, every node fails the handshake), so downloads
# go direct to the mirror with per-file retry and resume.
set -uo pipefail
REPO="${1:?repo}"
STEP="${2:?step}"
DEST="${3:?dest}"
BASE="https://hf-mirror.com/$REPO/resolve/main"
mkdir -p "$DEST"

echo "=== listing $REPO/$STEP ==="
HF_ENDPOINT=https://hf-mirror.com /home/leju-suzhou/miniconda3/envs/pi05/bin/python - "$REPO" "$STEP" > /tmp/filelist.txt <<'PY'
import os, sys
from huggingface_hub import HfApi
repo, step = sys.argv[1], sys.argv[2]
tok = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
api = HfApi()
info = api.model_info(repo, token=tok, files_metadata=True)
for s in (info.siblings or []):
    p = s.rfilename
    if p.startswith(step + "/") and (s.size or 0) > 0:
        print(f"{p}\t{s.size}")
PY
n=$(wc -l < /tmp/filelist.txt)
echo "  $n files"

fail=0
while IFS=$'\t' read -r path size; do
  [ -z "$path" ] && continue
  out="$DEST/$path"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$size" ]; then continue; fi
  ok=0
  for attempt in 1 2 3 4 5 6; do
    # Resume only when a partial file already exists; `-C -` on a missing file makes some
    # curl builds write nothing. Avoid an empty array expansion under `set -u`.
    if [ -s "$out" ]; then
      rc=0
      curl -sS -L --retry 3 --retry-delay 2 --retry-all-errors \
        --connect-timeout 20 --max-time 5400 \
        -C - -o "$out" "$BASE/$path" 2>/dev/null || rc=$?
    else
      rc=0
      curl -sS -L --retry 3 --retry-delay 2 --retry-all-errors \
        --connect-timeout 20 --max-time 5400 \
        -o "$out" "$BASE/$path" 2>/dev/null || rc=$?
    fi
    if [ "$(stat -c%s "$out" 2>/dev/null || echo 0)" = "$size" ]; then ok=1; break; fi
    sleep $((attempt * 2))
  done
  if [ "$ok" = 1 ]; then
    echo "  OK   $path"
  else
    echo "  FAIL $path (got $(stat -c%s "$out" 2>/dev/null || echo 0) / $size)"
    fail=$((fail+1))
  fi
done < /tmp/filelist.txt

echo "=== done: $((n-fail))/$n files ok, $fail failed ==="
du -sh "$DEST" 2>/dev/null
[ "$fail" -eq 0 ] && echo "DOWNLOAD_COMPLETE" || echo "DOWNLOAD_INCOMPLETE"
