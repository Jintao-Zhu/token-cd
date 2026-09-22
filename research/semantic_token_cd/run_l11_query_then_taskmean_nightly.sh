#!/usr/bin/env bash
set -uo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
cd "$WS"
echo "[$(date '+%F %T')] nightly query-provenance + taskmean chain start"
bash research/semantic_token_cd/run_l11_query_provenance_closed_loop.sh
S1=$?
if [[ "$S1" -ne 0 ]]; then echo "[$(date '+%F %T')] query provenance failed status=$S1"; exit "$S1"; fi
echo "[$(date '+%F %T')] query provenance complete; starting taskmean"
bash research/semantic_token_cd/run_l11_taskmean_closed_loop.sh
S2=$?
if [[ "$S2" -ne 0 ]]; then echo "[$(date '+%F %T')] taskmean failed status=$S2"; exit "$S2"; fi
echo "[$(date '+%F %T')] nightly chain complete"
