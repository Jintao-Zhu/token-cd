#!/usr/bin/env bash
# usage: run_embed_ab.sh <phys_gpu> <embed_jit 0|1> <tag>
set -u
B=/home/leju-suzhou/zjt_ws/token-cd
OV=/home/leju-suzhou/zjt_ws/pi05_sklearn_overlay
export OPENPI_DATA_HOME=/home/leju-suzhou/zjt_ws/checkpoints/openpi-assets
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$OV":/home/leju-suzhou/zjt_ws/openpi/src:/home/leju-suzhou/zjt_ws/openpi/packages/openpi-client/src:$B
export CUDA_VISIBLE_DEVICES="${1:-3}" BENCH_PHYS_GPU="${1:-3}" JAX_PLATFORMS=cuda,cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PI0FAST_CD_KMEANS=sklearn PI0FAST_CD_TAP_MODE=separate PI0FAST_CD_DECODE=verified
export PI0FAST_CD_BATCH=0 PI0FAST_CD_EMBED_JIT="${2:-0}"
cd /home/leju-suzhou/zjt_ws/openpi || exit 1
exec /home/leju-suzhou/miniconda3/envs/pi05/bin/python "$B/research/semantic_token_cd/pi0_fast_batch_validate.py" "${3:-run}"
