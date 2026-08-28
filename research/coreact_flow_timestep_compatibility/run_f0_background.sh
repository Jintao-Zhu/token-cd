#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/code
ARTIFACT="$1"
PYTHON="$ROOT/task1/.conda-envs/flow-vla/bin/python"
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1
export HF_HOME="$ROOT/artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522/hf_home"
export LIBERO_CONFIG_PATH="$ROOT/artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522/libero_config"
export PYTHONPATH="$ROOT/research/worktrees/lerobot_trained_weak_clean_e40b58a/src:$ROOT/LIBERO:$ROOT"
cd "$ROOT"

printf '{"stage":"SELECTION_RUNNING","states_planned":250}\n' > "$ARTIFACT/status/current.json"
"$PYTHON" -m research.coreact_flow_timestep_compatibility.run --workspace "$ROOT" --artifact "$ARTIFACT" --split selection > "$ARTIFACT/logs/selection.log" 2>&1
"$PYTHON" -m research.coreact_flow_timestep_compatibility.analyze --artifact "$ARTIFACT" --split selection > "$ARTIFACT/logs/selection_analysis.log" 2>&1

if [[ -f "$ARTIFACT/selected_segment.lock.json" ]]; then
  printf '{"stage":"CONFIRMATION_RUNNING","states_planned":250}\n' > "$ARTIFACT/status/current.json"
  "$PYTHON" -m research.coreact_flow_timestep_compatibility.run --workspace "$ROOT" --artifact "$ARTIFACT" --split confirmation > "$ARTIFACT/logs/confirmation.log" 2>&1
  "$PYTHON" -m research.coreact_flow_timestep_compatibility.analyze --artifact "$ARTIFACT" --split confirmation > "$ARTIFACT/logs/confirmation_analysis.log" 2>&1
fi

decision=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "$ARTIFACT/decision.json")
printf '{"stage":"COMPLETE","decision":"%s"}\n' "$decision" > "$ARTIFACT/status/current.json"
