#!/usr/bin/env bash
set -euo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
ROOT="$WS/artifacts/l11_matched_dtp_positive_lambda_060_075_v1"
slot="${1:?slot required}"
mkdir -p "$ROOT/logs"
while [ ! -f "$ROOT/preflight/PASS" ]; do sleep 10; done
attempt=0
while true; do
  if bash "$WS/research/semantic_token_cd/run_l11_high_lambda1600.sh" "$slot"; then exit 0; fi
  attempt=$((attempt + 1))
  echo "[$(date --iso-8601=seconds)] shard=$slot retry=$attempt" >&2
  sleep 30
done

