#!/usr/bin/env bash
# Crash-isolated close_drawer replay driver: each seed runs in a FRESH process
# (SAPIEN 2.2.2 intermittently heap-corrupts after ~5k steps, so one process per
# seed keeps every run far below the trigger). Re-runs until all seeds are done.
set -u
cd ~/zjt_ws/token-cd
OUT=artifacts/attn_semantic_merge_k8_v1/replay_close_drawer
PY=~/zjt_ws/openvla-ar-h100/bin/python
SEEDS="200 201 202 205 206 207 209 210 211 213 214 215 216 217 220 221 222 223 224 226 227 233 236 241 248 253 258 260 270 271 277 281 286 287 294 297"
GPUS=(1 4 5 6 7)

run_one() {
  s=$1; gpu=$2
  f="$OUT/seed_$(printf %03d $s).json"
  [ -f "$f" ] && return 0
  timeout 400 env CUDA_VISIBLE_DEVICES=$gpu "$PY" \
    research/semantic_token_cd/replay_close_drawer.py --gpu "$gpu" --seeds "$s" --out "$OUT" \
    > "/tmp/replay_seed_$s.log" 2>&1
}

for round in $(seq 1 40); do
  pending=""
  for s in $SEEDS; do
    [ -f "$OUT/seed_$(printf %03d $s).json" ] || pending="$pending $s"
  done
  [ -z "$pending" ] && { echo "ALL_DONE round=$round"; break; }
  echo "round $round pending: $pending"
  i=0
  for s in $pending; do
    gpu=${GPUS[$((i % 5))]}
    run_one "$s" "$gpu" &
    i=$((i+1))
    [ $((i % 5)) -eq 0 ] && wait
  done
  wait
done
echo "FINAL done=$(ls "$OUT"/seed_*.json 2>/dev/null | wc -l)/$(echo $SEEDS | wc -w)"
