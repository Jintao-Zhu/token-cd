#!/usr/bin/env bash
set -u
ROOT=/home/leju-suzhou/zjt_ws
source /home/leju-suzhou/miniconda3/etc/profile.d/conda.sh
conda activate pi0
export CUDA_VISIBLE_DEVICES="${1:-1}"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export DET_TASK="${2:-widowx_carrot_on_plate}" DET_SEED="${3:-0}" DET_RT_DEPTH="${4:-8}"
export PYTHONPATH="$ROOT/openpi/packages/openpi-client/src:$ROOT/pcd_openvla_simpler_box_31b027e/source:$ROOT/SimplerEnv:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/SpatialVLA/ManiSkill2_real2sim:$ROOT/token-cd:$ROOT/openpi/src"
exec /home/leju-suzhou/miniconda3/envs/pi0/bin/python "$ROOT/token-cd/scripts/check_widowx_env.py"
