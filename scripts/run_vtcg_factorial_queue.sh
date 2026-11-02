#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/leju-suzhou/zjt_ws/token-cd
source "$ROOT/scripts/activate_simpler_local.sh"
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Queue CPU process does not initialize CUDA. It sets one authorized physical GPU
# explicitly for each subprocess. No worker may inherit an unrestricted device set.
export CUDA_VISIBLE_DEVICES=''
exec "$SIMPLER_ENV/bin/python" -u "$ROOT/scripts/vtcg_factorial_queue.py" \
 --artifact "$ROOT/artifacts/vtcg_factorial_controls_v1" "$@"
