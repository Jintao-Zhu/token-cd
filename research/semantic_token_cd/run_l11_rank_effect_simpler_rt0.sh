#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?usage: run_l11_rank_effect_simpler_rt0.sh GPU ARTIFACT [SMOKE] [TASKS]}"
ARTIFACT="${2:?missing artifact directory}"
MODE="${3:-smoke}"
TASKS="${4:-google_robot_open_drawer,google_robot_close_drawer,google_robot_pick_coke_can,google_robot_move_near}"
SEEDS="${5:-100-129}"
STATES="${6:-10}"
case "$GPU" in 1|2|3|4|5|6) ;; *) echo "inference GPU $GPU is not allowed" >&2; exit 2 ;; esac
WS=/home/leju-suzhou/zjt_ws/token-cd
PCD=/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
cd "$WS"
export CUDA_VISIBLE_DEVICES="$GPU,7"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTHONPATH="$WS:$PCD:$WS/task1/shim_site${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ARTIFACT/logs"
ARGS=(--gpu "$GPU" --worker-id "rt0-gpu${GPU}" --artifact "$ARTIFACT" \
  --states-per-episode "$STATES" --seeds "$SEEDS" --tasks "$TASKS")
if [ "$MODE" = "smoke" ]; then ARGS+=(--smoke); fi
/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python \
  research/semantic_token_cd/replay_simpler_l11_rank_effects.py "${ARGS[@]}"
