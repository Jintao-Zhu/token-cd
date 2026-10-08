#!/usr/bin/env bash
set -euo pipefail

gpu_id="$1"
task_id="$2"
seed_spec="$3"
worker_id="$4"
repo_root="/home/leju-suzhou/zjt_ws/token-cd"
python_bin="/home/leju-suzhou/miniconda3/envs/pi0/bin/python"
artifact_dir="$repo_root/artifacts/pi0_l10_token_pcd_remaining5_rtfull_0_99_v1"
log_dir="$artifact_dir/logs"
mkdir -p "$log_dir"
export CUDA_VISIBLE_DEVICES="$gpu_id"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
cd "$repo_root"
exec "$python_bin" -u research/semantic_token_cd/pi0_l10_pcd_remaining5_rollout.py \
  --task "$task_id" --seeds "$seed_spec" --gpu "$gpu_id" \
  --worker-id "$worker_id" --artifact "$artifact_dir" \
  >"$log_dir/$worker_id.log" 2>&1
