#!/usr/bin/env bash
# Replan-cost benchmark launcher. Usage: run_bench.sh <phys_gpu> <tap_mode> <decode_mode> <tag>
set -u
GPU="${1:?gpu}"
TAP="${2:?tap_mode}"
DEC="${3:?decode_mode}"
TAG="${4:?tag}"
B=/home/leju-suzhou/zjt_ws/token-cd
export OPENPI_DATA_HOME=/home/leju-suzhou/zjt_ws/checkpoints/openpi-assets
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/home/leju-suzhou/zjt_ws/openpi/src:/home/leju-suzhou/zjt_ws/openpi/packages/openpi-client/src:$B
export CUDA_VISIBLE_DEVICES="$GPU" BENCH_PHYS_GPU="$GPU" JAX_PLATFORMS=cuda,cpu
export PI0FAST_TIMING=1 PI0FAST_CD_TAP_MODE="$TAP" PI0FAST_CD_DECODE="$DEC"
cd /home/leju-suzhou/zjt_ws/openpi || exit 1
exec /home/leju-suzhou/miniconda3/envs/pi05/bin/python \
  "$B/research/semantic_token_cd/pi0_fast_replan_timing.py"
