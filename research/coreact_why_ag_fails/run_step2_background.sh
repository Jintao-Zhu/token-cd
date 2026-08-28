#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/code
ARTIFACT="$1"
SOURCE="$ROOT/artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522"
PYTHON="$ROOT/task1/.conda-envs/flow-vla/bin/python"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl HF_HUB_OFFLINE=1 HF_HOME="$SOURCE/hf_home" LIBERO_CONFIG_PATH="$SOURCE/libero_config"
export PYTHONPATH="$ROOT/research/worktrees/lerobot_trained_weak_clean_e40b58a/src:$ROOT/LIBERO:$ROOT"
cd "$ROOT"
printf '{"stage":"ACTION_MANIFOLD_RUNNING","states":250}\n' > "$ARTIFACT/status/current.json"
"$PYTHON" -m research.coreact_why_ag_fails.run_manifold_diagnosis --workspace "$ROOT" --artifact "$ARTIFACT" > "$ARTIFACT/logs/run.log" 2>&1
"$PYTHON" -m research.coreact_why_ag_fails.analyze_manifold_diagnosis --artifact "$ARTIFACT" > "$ARTIFACT/logs/analysis.log" 2>&1
