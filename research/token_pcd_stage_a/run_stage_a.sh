#!/usr/bin/env bash
set -euo pipefail

workspace=/root/code
pcd_root="$workspace/official-reproductions/pcd_openvla_simpler_box_31b027e"
artifact="${1:?usage: run_stage_a.sh ARTIFACT_PATH}"
python_bin="$pcd_root/env/venv/bin/python"

cd "$workspace"
export PYTHONPATH="$workspace:$pcd_root/source/PCD"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYOPENGL_PLATFORM=egl
export DISPLAY=

"$python_bin" -m research.token_pcd_stage_a.capture_sam_masks --pcd-root "$pcd_root" --artifact "$artifact"
"$python_bin" -m research.token_pcd_stage_a.run --pcd-root "$pcd_root" --artifact "$artifact" --preflight-only
"$python_bin" -m research.token_pcd_stage_a.run --pcd-root "$pcd_root" --artifact "$artifact" --integrity-only
"$python_bin" -m research.token_pcd_stage_a.run --pcd-root "$pcd_root" --artifact "$artifact"
"$python_bin" -m research.token_pcd_stage_a.analyze --artifact "$artifact"
"$python_bin" -m research.token_pcd_stage_a.finalize --artifact "$artifact" --workspace "$workspace"
