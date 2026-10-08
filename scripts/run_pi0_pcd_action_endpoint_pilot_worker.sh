#!/usr/bin/env bash
set -euo pipefail

gpu_id="$1"
task_id="$2"
worker_id="$3"
preflight_flag="${4:-}"
seed_spec="${5:-}"
repo_root="/home/leju-suzhou/zjt_ws/token-cd"
python_bin="/home/leju-suzhou/miniconda3/envs/pi0/bin/python"
artifact_dir="$repo_root/artifacts/pi0_action_endpoint_pilot_v1"
log_dir="$artifact_dir/logs"
mkdir -p "$log_dir"
export CUDA_VISIBLE_DEVICES="$gpu_id"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
cd "$repo_root"
args=(--task "$task_id" --gpu "$gpu_id" --worker-id "$worker_id" --artifact "$artifact_dir")
if [[ "$preflight_flag" == "preflight" ]]; then
  args+=(--preflight-only)
fi
if [[ -n "$seed_spec" ]]; then
  args+=(--seeds "$seed_spec")
fi
exec "$python_bin" -u research/semantic_token_cd/pi0_pcd_action_endpoint_pilot.py "${args[@]}" \
  >"$log_dir/$worker_id.log" 2>&1
