#!/usr/bin/env bash
set -u
B=/home/leju-suzhou/zjt_ws/token-cd
OV=/home/leju-suzhou/zjt_ws/pi05_sklearn_overlay
export OPENPI_DATA_HOME=/home/leju-suzhou/zjt_ws/checkpoints/openpi-assets
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$OV":/home/leju-suzhou/zjt_ws/openpi/src:/home/leju-suzhou/zjt_ws/openpi/packages/openpi-client/src:$B
export CUDA_VISIBLE_DEVICES="${1:-1}" BENCH_PHYS_GPU="${1:-1}" JAX_PLATFORMS=cuda,cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /home/leju-suzhou/zjt_ws/openpi || exit 1
exec /home/leju-suzhou/miniconda3/envs/pi05/bin/python "$B/research/semantic_token_cd/pi0_fast_kmeans_backend.py"
