#!/usr/bin/env bash
# Watchdog: when the full coordinator reports ALL_JOBS_FINISHED, run the
# offline same-state decode and then the aggregate analyzer. Kept in tmux.
set -u
cd /home/leju-suzhou/zjt_ws/token-cd
export PYTHONPATH=/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
LOG=logs/xswap

echo "[watchdog] waiting for rollout ALL_JOBS_FINISHED at $(date '+%F %T')"
for i in $(seq 1 720); do
  grep -q 'ALL_JOBS_FINISHED' "$LOG/coordinator_main.log" 2>/dev/null && break
  sleep 30
done
if ! grep -q 'ALL_JOBS_FINISHED' "$LOG/coordinator_main.log" 2>/dev/null; then
  echo "[watchdog] rollout did not finish within timeout; continuing anyway" >> "$LOG/watchdog.log"
fi
echo "[watchdog] rollout finished $(date '+%T'); launching offline decode"
"$PY" research/semantic_token_cd/xswap_offline_coordinator.py >> "$LOG/offline_main.log" 2>&1
echo "[watchdog] offline decode done $(date '+%T')"
"$PY" research/semantic_token_cd/xswap_analyze.py --out artifacts/prompt_attn_instr_swap_v1/REPORT.md \
  >> "$LOG/analyze.log" 2>&1
echo "[watchdog] analysis done $(date '+%T') rc=$?"
echo ALL_PHASES_DONE >> "$LOG/watchdog.log"
echo "[watchdog] ALL_PHASES_DONE at $(date '+%F %T')"
