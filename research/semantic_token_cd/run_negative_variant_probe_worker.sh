#!/usr/bin/env bash
set -uo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
OUT="$WS/artifacts/libero_transfer_diagnostics/negative_variant_probe_v4"
gpu="${1:?usage: run_negative_variant_probe_worker.sh GPU_ID}"
renderer="${2:-$gpu}"
case "$gpu" in 1|3|6|7) ;; *) echo "GPU not authorized for this probe: $gpu" >&2; exit 2 ;; esac
case "$renderer" in 1|3|6|7) ;; *) echo "renderer not authorized for this probe: $renderer" >&2; exit 2 ;; esac
cd "$WS"
source scripts/activate_libero_openvla.sh
set +e
if [ "$gpu" = "$renderer" ]; then
  export CUDA_VISIBLE_DEVICES="$gpu"
else
  export CUDA_VISIBLE_DEVICES="$gpu,$renderer"
fi
export MUJOCO_EGL_DEVICE_ID="$renderer"
export TOKENIZERS_PARALLELISM=false
while IFS= read -r case_id; do
  [ -n "$case_id" ] || continue
  log="$OUT/logs/gpu${gpu}_${case_id}.log"
  mkdir -p "$OUT/logs"
  python research/semantic_token_cd/collect_negative_variant_probe.py \
    --gpu "$gpu" --device-index 0 --case-id "$case_id" --out "$OUT" \
    > "$log" 2>&1
  status=$?
  case_dir="$OUT/$case_id"
  if [ "$status" -ne 0 ] && { [ ! -f "$case_dir/metadata.json" ] || [ ! -f "$case_dir/margin_logits.npz" ]; }; then
    echo "failed case=$case_id exit=$status"
    tail -40 "$log"
    exit "$status"
  fi
  echo "finished case=$case_id exit=$status"
done < "$OUT/gpu${gpu}.txt"
