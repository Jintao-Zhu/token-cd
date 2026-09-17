#!/usr/bin/env bash
# After the rollout coordinator reports ALL_JOBS_FINISHED:
#  1) offline decode for any task still missing (pick_coke), 2) final REPORT.
set -u
cd /home/leju-suzhou/zjt_ws/token-cd
export PYTHONPATH=/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
LOG=logs/xswap
echo "[finalize] waiting for rollout completion at $(date '+%F %T')"
for i in $(seq 1 240); do
  grep -q 'ALL_JOBS_FINISHED' "$LOG/coordinator_main.log" && break
  sleep 30
done
sleep 60   # let last shard flush arrays/summaries
echo "[finalize] rollout done $(date '+%T'); waiting for offline-r2 wave"
for i in $(seq 1 180); do
  grep -q 'OFFLINE_R2_ALL_DONE' "$LOG/offline_r2_gpu3.log" && break
  sleep 20
done
echo "[finalize] offline-r2 done $(date '+%T'); offline decode (skips done tasks)"
"$PY" research/semantic_token_cd/xswap_offline_coordinator.py >> "$LOG/finalize_offline.log" 2>&1
echo "[finalize] offline rc=$? $(date '+%T'); running final analysis"
"$PY" research/semantic_token_cd/xswap_analyze.py --out artifacts/prompt_attn_instr_swap_v1/REPORT.md \
   >> "$LOG/finalize_analyze.log" 2>&1
echo "[finalize] analysis rc=$? $(date '+%T')"
echo FINALIZE_ALL_DONE >> "$LOG/finalize.log"
