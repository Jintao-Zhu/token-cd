#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/code;ART="$1";SOURCE="$ROOT/artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522";PY="$ROOT/task1/.conda-envs/flow-vla/bin/python"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl HF_HUB_OFFLINE=1 HF_HOME="$SOURCE/hf_home" LIBERO_CONFIG_PATH="$SOURCE/libero_config" PYTHONPATH="$ROOT/research/worktrees/lerobot_trained_weak_clean_e40b58a/src:$ROOT/LIBERO:$ROOT"
cd "$ROOT";printf '{"stage":"CONFIRMATION_RUNNING"}\n' > "$ART/status/current.json"
"$PY" -m research.coreact_why_ag_fails.run_self_weak_confirmation --workspace "$ROOT" --artifact "$ART" > "$ART/logs/confirmation_run.log" 2>&1
"$PY" -m research.coreact_why_ag_fails.analyze_self_weak_confirmation --artifact "$ART" > "$ART/logs/confirmation_analysis.log" 2>&1
