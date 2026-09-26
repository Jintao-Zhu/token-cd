#!/usr/bin/env bash
set -uo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
ROOT="$WS/artifacts/libero_two_bowl_attention_localization_v1_20260926"
gpu="$1"
arm="$2"
task="$3"
episodes="$4"
wid="$5"
source "$WS/scripts/activate_libero_openvla.sh"
export CUDA_VISIBLE_DEVICES="7,$gpu"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs/anchor" "$ROOT/episodes/$arm"
log="$ROOT/logs/anchor/${wid}.log"
out="$ROOT/episodes/$arm"
echo "worker=$wid gpu=$gpu arm=$arm task=$task episodes=$episodes render_gpu=7 start=$(date '+%F %T')" > "$log"
python "$WS/research/semantic_token_cd/libero_two_bowl_anchor_rollout.py" \
  --gpu "$gpu" --render-gpu 7 --arm "$arm" --task "$task" \
  --episodes "$episodes" --artifact "$out" \
  >> "$log" 2>&1
rc=$?
echo "exit_code=$rc end=$(date '+%F %T')" >> "$log"
exit "$rc"
