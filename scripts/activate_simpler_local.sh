#!/usr/bin/env bash
# Local SIMPLER/OpenVLA runtime for this checkout (no Docker dependency).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "$REPO_ROOT/.." && pwd)"

export TOKEN_CD="${TOKEN_CD:-$REPO_ROOT}"
export PCD_ROOT="${PCD_ROOT:-$WORKSPACE_ROOT/pcd_openvla_simpler_box_31b027e}"
export PCD_SOURCE="${PCD_SOURCE:-$PCD_ROOT/source}"
export SIMPLER_ENV="${SIMPLER_ENV:-$WORKSPACE_ROOT/openvla-ar-h100}"
export OPENVLA_BASE_CHECKPOINT="${OPENVLA_BASE_CHECKPOINT:-$PCD_SOURCE/pretrained/openvla-7b}"

export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$TOKEN_CD/task1/shim_site:$TOKEN_CD:$PCD_SOURCE:$TOKEN_CD/third_party/openvla${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$SIMPLER_ENV/bin:$PATH"

for required in "$SIMPLER_ENV/bin/python" "$PCD_SOURCE/pretrained/openvla-7b/config.json" \
                "$TOKEN_CD/research/semantic_token_cd/simpler_geometry_mask_rollout.py"; do
  if [[ ! -e "$required" ]]; then
    echo "Required local SIMPLER path is missing: $required" >&2
    return 1 2>/dev/null || exit 1
  fi
done
