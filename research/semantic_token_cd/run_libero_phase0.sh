#!/usr/bin/env bash
# Driver for LIBERO_OBJECT_SEMANTIC_ENTITY_CD_PHASE0_V1 — offline alignment gate.
set -euo pipefail

cd /home/leju-suzhou/zjt_ws/token-cd

GPU="${GPU:-0}"
SMOKE="${SMOKE:-0}"
ARTIFACT="artifacts/libero_object_semantic_entity_cd_phase0_v1"

export HF_HUB_OFFLINE=1
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=1
export PYTHONPATH="./LIBERO:$PWD"

ARGS=(--artifact "$ARTIFACT" --gpu "$GPU")
if [[ "$SMOKE" == "1" ]]; then
  ARGS+=(--smoke)
fi

exec task1/.venvs/openvla-ar/bin/python -u research/semantic_token_cd/libero_phase0_align.py "${ARGS[@]}"
