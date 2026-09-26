#!/usr/bin/env bash
set -uo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
OUT="$WS/artifacts/libero_transfer_diagnostics/libero_mild_view_logits"
cd "$WS"
source scripts/activate_libero_openvla.sh
set +e
export CUDA_VISIBLE_DEVICES=1,2
export MUJOCO_EGL_DEVICE_ID=1
export TOKENIZERS_PARALLELISM=false
manifest="$OUT/manifests/gpu1.txt"
while IFS= read -r case_id; do
  [ -n "$case_id" ] || continue
  case_dir="$OUT/$case_id"
  if [ -f "$case_dir/metadata.json" ] && [ -f "$case_dir/action_logits.npz" ]; then
    echo "skip-existing $case_id"
    continue
  fi
  echo "start $case_id"
  python research/semantic_token_cd/collect_libero_mild_view_stability.py \
    --gpu 1 --device-index 0 --case-ids "$case_id" --skip-existing \
    --rgb-source live-replay --out "$OUT" \
    > "$OUT/logs/gpu1_case_${case_id}.log" 2>&1
  status=$?
  if [ "$status" -ne 0 ] && { [ ! -f "$case_dir/metadata.json" ] || [ ! -f "$case_dir/action_logits.npz" ]; }; then
    echo "failed $case_id exit=$status"
    tail -40 "$OUT/logs/gpu1_case_${case_id}.log"
    exit "$status"
  fi
  echo "finished $case_id exit=$status"
done < <(tr ', ' '\n' < "$manifest" | sed '/^$/d')
