#!/usr/bin/env bash
set -uo pipefail

cd /home/leju-suzhou/zjt_ws/token-cd
SOURCE=artifacts/projected_shr_5task_0_299_v1/logs/COMPLETE
LOG=artifacts/pi0_shr_5task_0_299_v1/logs/chain.log
BRIDGE=/home/leju-suzhou/zjt_ws/checkpoints/pi0/bridge_beta_step19296_2024-12-26_22-30_42.pt
BRIDGE_SIZE=11773024888
mkdir -p "$(dirname "$LOG")"

printf '%s waiting for %s\n' "$(date --iso-8601=seconds)" "$SOURCE" >> "$LOG"
while [ ! -f "$SOURCE" ]; do
  sleep 30
done
printf '%s projected experiment complete; checking Bridge-Beta checkpoint\n' \
  "$(date --iso-8601=seconds)" >> "$LOG"
# aria2 creates a sparse, final-size output file before it has downloaded all
# chunks, so wait for its sidecar to disappear as well.
while [ ! -f "$BRIDGE" ] || [ -e "$BRIDGE.aria2" ] || [ "$(stat -c %s "$BRIDGE" 2>/dev/null || printf 0)" -ne "$BRIDGE_SIZE" ]; do
  sleep 30
done
printf '%s projected experiment complete; starting Pi0-SHR\n' \
  "$(date --iso-8601=seconds)" >> "$LOG"
exec bash research/semantic_token_cd/run_pi0_shr_5task_0_299_gpu145.sh >> "$LOG" 2>&1
