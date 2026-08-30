#!/usr/bin/env bash
# SCR-CD Phase 1 — GPU 0 re-crashed (Xid 45 -> 154 "GPU Reset Required").
# Re-dispatch the 6 jobs that were assigned to GPU 0 onto healthy GPUs {1,4,5,6,7},
# adding at most 1 extra worker per GPU (OOM-safe; H100 ~81GB, primary workers
# already ~48-55GB). Idempotent: skip-if-.done. Writes the SAME .done markers the
# orchestrator's RETRY pass checks, so RETRY skips these instead of hitting dead GPU 0.
#
# Wave 1 (5 concurrent, 1 extra per healthy GPU), then wave 2 (6th job).
set -u
cd ~/zjt_ws/token-cd

PY=~/zjt_ws/openvla-ar-h100/bin/python
SCD=research/semantic_token_cd
ART=artifacts/semantic_recon_k8_m10_v1
PP=~/zjt_ws/token-cd:~/zjt_ws/pcd_openvla_simpler_box_31b027e/source
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# "task|shard|idx|gpu" — idx must match the orchestrator's job index so the
# .done marker path (rollout_<short>_s<idx>.log.done) lines up exactly.
JOBS=(
  "google_robot_move_near|300-319|15|1"
  "google_robot_move_near|320-339|16|4"
  "google_robot_move_near|340-359|17|5"
  "widowx_put_eggplant_in_basket|360-379|33|6"
  "widowx_put_eggplant_in_basket|380-399|34|7"
  "widowx_spoon_on_towel|300-319|35|1"
)

shortname() { echo "$1" | sed 's/google_robot_//'; }

run_one() {
  local task=$1 shard=$2 idx=$3 gpu=$4
  local short; short=$(shortname "$task")
  local log="$ART/logs/rollout_${short}_s${idx}.log"
  if [ -f "$log.done" ]; then
    echo "[remediate] skip done idx=$idx $task $shard"
    return 0
  fi
  # Preserve the GPU-0 crash trace, then start a fresh log.
  if [ -f "$log" ] && [ ! -f "$log.crashed" ]; then
    mv "$log" "$log.crashed"
  fi
  env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$PP" "$PY" \
    "$SCD/semantic_recon_rollout.py" --artifact "$ART" --task "$task" --gpu "$gpu" \
    --seeds "$shard" > "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then touch "$log.done"; echo "[remediate] DONE idx=$idx $task $shard gpu=$gpu"; else
    echo "[remediate] FAILED rc=$rc idx=$idx $task $shard gpu=$gpu (log: $log)"
  fi
  return $rc
}

echo "=== remediation wave 1 (5 concurrent) $(date +%H:%M:%S) ==="
for i in 0 1 2 3 4; do
  IFS='|' read -r task shard idx gpu <<< "${JOBS[$i]}"
  run_one "$task" "$shard" "$idx" "$gpu" &
done
wait

echo "=== remediation wave 2 (6th job) $(date +%H:%M:%S) ==="
IFS='|' read -r task shard idx gpu <<< "${JOBS[5]}"
run_one "$task" "$shard" "$idx" "$gpu"

echo "=== remediation complete $(date +%H:%M:%S) ==="
