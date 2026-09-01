#!/usr/bin/env bash
# Monitor for Phase 4 gap-fill backfill (6 tasks x 0-299 x 2 arms = 3600 episodes).
set -u
cd ~/zjt_ws/token-cd
ART=artifacts/semantic_recon_k8_m10_0_299_baseline_backfill_v1

count() {
  ls -1 "$ART"/episodes/*/*/episode_*.json 2>/dev/null | wc -l
}
done_shards() {
  ls -1 "$ART"/logs/backfill_*.log.done 2>/dev/null | wc -l
}

TARGET=3600
MAX_MIN=$((60 * 20))   # 20h hard cap
START=$(date +%s)

while :; do
  r=$(count)
  d=$(done_shards)
  now=$(date +%s)
  el=$(( (now - START) / 60 ))
  rate=$(( r / (el + 1) ))
  echo "[monitor] +${el}min episodes=$r/$TARGET shards=$d/90 (~${rate}ep/min)"
  if [ "$r" -ge "$TARGET" ] && [ "$d" -ge 90 ]; then
    echo "[monitor] COMPLETE episodes=$r shards=$d"
    exit 0
  fi
  if [ "$el" -ge "$MAX_MIN" ]; then
    echo "[monitor] TIMEOUT at ${el}min (episodes=$r shards=$d)"
    exit 2
  fi
  sleep 300
done
