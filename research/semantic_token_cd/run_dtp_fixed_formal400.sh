#!/usr/bin/env bash
# DTP fixed-mask formal 400. GPU3 starts immediately; GPU2 workers wait for P75.
set -u

DTP_WS=/home/leju-suzhou/zjt_ws/token-cd
DTP_PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
DTP_ROOT="$DTP_WS/artifacts/dtp_openvla_calibration_v1/formal_fixed400"
DTP_ARM=v2_fix_k64_t05
mkdir -p "$DTP_ROOT/logs"

run_chunk() {
  local worker=$1 gpu=$2 first_task=$3 first_seeds=$4 second_task=$5 second_seeds=$6
  export PYTHONPATH="$DTP_WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
  export HF_HUB_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false
  cd "$DTP_WS" || exit 1
  "$DTP_PY" research/semantic_token_cd/dtp_closed_loop_rollout.py \
    --task "$first_task" --seeds "$first_seeds" --arms "$DTP_ARM" --gpu "$gpu" \
    --worker-id "$worker" --artifact-dir "$DTP_ROOT"
  "$DTP_PY" research/semantic_token_cd/dtp_closed_loop_rollout.py \
    --task "$second_task" --seeds "$second_seeds" --arms "$DTP_ARM" --gpu "$gpu" \
    --worker-id "$worker" --artifact-dir "$DTP_ROOT"
}

wait_for_p75_gpu2() {
  while pgrep -f 'vla_pruner_p75_rollout.*--gpu 2' >/dev/null; do
    sleep 20
  done
}

case "${1:-}" in
  g3a) run_chunk g3a 3 google_robot_open_drawer 0-49 google_robot_pick_coke_can 0-49 ;;
  g3b) run_chunk g3b 3 google_robot_close_drawer 0-49 google_robot_move_near 0-49 ;;
  g2a) wait_for_p75_gpu2; run_chunk g2a 2 google_robot_open_drawer 50-99 google_robot_pick_coke_can 50-99 ;;
  g2b) wait_for_p75_gpu2; run_chunk g2b 2 google_robot_close_drawer 50-99 google_robot_move_near 50-99 ;;
  finalize)
    while [ "$(find "$DTP_ROOT/episodes" -type f -path "*/$DTP_ARM/episode_*_summary.json" 2>/dev/null | wc -l)" -lt 400 ]; do
      sleep 30
    done
    cd "$DTP_WS" || exit 1
    "$DTP_PY" research/semantic_token_cd/dtp_fixed_formal_analyze.py \
      > "$DTP_ROOT/logs/finalizer_analysis.json"
    ;;
  *) echo "usage: $0 {g3a|g3b|g2a|g2b|finalize}" >&2; exit 2 ;;
esac
