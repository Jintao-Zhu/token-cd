#!/usr/bin/env bash
set -euo pipefail

workspace=/data/docker/dev_zjt/data/code
artifact="${1:?usage: run_background.sh ARTIFACT}"
pcd_root="$workspace/official-reproductions/pcd_openvla_simpler_box_31b027e"
python_bin="$pcd_root/env/venv/bin/python"

cd "$workspace"
export PYTHONPATH="$workspace:$pcd_root/source/PCD:$pcd_root/runner"
export CUDA_VISIBLE_DEVICES=0
export DISPLAY=
export PYOPENGL_PLATFORM=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$artifact/logs"
nohup "$python_bin" -u -m research.token_pcd_bridge.supervisor \
  --workspace "$workspace" --artifact "$artifact" \
  > "$artifact/logs/supervisor.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$artifact/supervisor.pid"
disown "$pid" 2>/dev/null || true
printf 'STARTED pid=%s artifact=%s\n' "$pid" "$artifact"
