#!/usr/bin/env bash
set -u
WS=/home/leju-suzhou/zjt_ws/token-cd
ROOT="$WS/artifacts/prompt_attn_l11_matched_dense_lambda_remaining5_0_99_v1"
slot="${1:?slot required}"
mkdir -p "$ROOT/logs"
attempt=0
while true; do
  if bash "$WS/research/semantic_token_cd/run_l11_dense_lambda_remaining5.sh" "$slot"; then exit 0; fi
  attempt=$((attempt + 1))
  echo "[$(date --iso-8601=seconds)] shard=$slot retry=$attempt" >&2
  sleep 30
done
