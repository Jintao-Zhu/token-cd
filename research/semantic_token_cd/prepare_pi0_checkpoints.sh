#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/leju-suzhou/zjt_ws/checkpoints/pi0
NAME=bridge_beta_step19296_2024-12-26_22-30_42.pt
TARGET="$ROOT/$NAME"
EXPECTED=11773024888
HFD=/home/leju-suzhou/zjt_ws/scripts/hfd.sh
ARIA_BIN=/home/leju-suzhou/miniconda3/envs/pi0/bin

mkdir -p "$ROOT"
checkpoint_ready() {
  [ -f "$TARGET" ] &&
    [ ! -e "$TARGET.aria2" ] &&
    [ "$(stat -c %s "$TARGET")" -eq "$EXPECTED" ]
}

# aria2 preallocates the sparse output file to its final logical size.  Its
# sidecar is therefore the completion signal; size alone would start π0 with
# an incomplete checkpoint.
if checkpoint_ready; then
  printf 'Bridge-Beta checkpoint already complete: %s\n' "$TARGET"
  exit 0
fi
[ -x "$HFD" ] || { echo "missing Hugging Face downloader: $HFD" >&2; exit 1; }
while ! checkpoint_ready; do
  HF_ENDPOINT=https://hf-mirror.com \
  HTTP_PROXY=http://127.0.0.1:7890 \
  HTTPS_PROXY=http://127.0.0.1:7890 \
  ALL_PROXY=http://127.0.0.1:7890 \
  http_proxy=http://127.0.0.1:7890 \
  https_proxy=http://127.0.0.1:7890 \
  all_proxy=http://127.0.0.1:7890 \
  PATH="$ARIA_BIN:$PATH" timeout 180 "$HFD" allenzren/open-pi-zero \
    --include "$NAME" --tool aria2c -x 10 -j 1 --local-dir "$ROOT" || true
  checkpoint_ready && break
  printf '%s retrying incomplete Bridge-Beta download\n' "$(date --iso-8601=seconds)" >&2
  sleep 5
done
checkpoint_ready || {
  echo "Bridge-Beta checkpoint size mismatch" >&2
  exit 1
}
printf 'Bridge-Beta checkpoint ready: %s (%s bytes)\n' "$TARGET" "$EXPECTED"
