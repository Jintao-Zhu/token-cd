#!/usr/bin/env bash
# SCR-CD Phase 2 (replication): backfill vanilla + semantic_attn for move_near 0-99.
# 5 shards (20 seeds each) x 2 arms = 200 episodes. One extra worker per healthy GPU
# (parallel with the recon rollout, ~17GB/worker, leaves ~10GB headroom per GPU).
set -u
cd ~/zjt_ws/token-cd

PY=~/zjt_ws/openvla-ar-h100/bin/python
SCD=research/semantic_token_cd
ART=artifacts/semantic_recon_0_199_core3_replication_v1
PP=~/zjt_ws/token-cd:~/zjt_ws/pcd_openvla_simpler_box_31b027e/source
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# shard -> gpu (one per healthy GPU)
JOBS=(
  "0-19|1"
  "20-39|4"
  "40-59|5"
  "60-79|6"
  "80-99|7"
)

mkdir -p "$ART/logs" "$ART/episodes"

run_job() {
  local shard=$1 gpu=$2
  local log="$ART/logs/backfill_move_near_${shard}.log"
  if [ -f "$log.done" ]; then
    echo "[backfill] skip done $shard"
    return 0
  fi
  env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$PP" "$PY" \
    "$SCD/semantic_recon_0_199_backfill.py" --artifact "$ART" --gpu "$gpu" \
    --seeds "$shard" > "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then touch "$log.done"; else
    echo "[backfill] FAILED rc=$rc shard=$shard gpu=$gpu (log: $log)"
  fi
  return $rc
}

echo "=== backfill dispatch (5 shards, 1/GPU) $(date +%H:%M:%S) ==="
for entry in "${JOBS[@]}"; do
  IFS='|' read -r shard gpu <<< "$entry"
  run_job "$shard" "$gpu" &
done
wait

# RETRY pass.
for attempt in 1 2 3; do
  MISSING=()
  for entry in "${JOBS[@]}"; do
    IFS='|' read -r shard gpu <<< "$entry"
    [ -f "$ART/logs/backfill_move_near_${shard}.log.done" ] || MISSING+=("$entry")
  done
  if [ ${#MISSING[@]} -eq 0 ]; then
    echo "=== BACKFILL ALL COMPLETE (attempt $attempt) ==="
    break
  fi
  echo "=== backfill retry $attempt: ${#MISSING[@]} incomplete ==="
  for entry in "${MISSING[@]}"; do
    IFS='|' read -r shard gpu <<< "$entry"
    run_job "$shard" "$gpu"
  done
done

echo "=== backfill complete $(date +%H:%M:%S) ==="
