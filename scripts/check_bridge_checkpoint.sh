#!/usr/bin/env bash
# One-off check that the bridge checkpoint makes the widowx robot actually interact.
# Starts a single bridge server plus a single rollout worker for one episode and prints the
# final_info interaction flags, which is what distinguishes "hard task" from "policy cannot
# act" (the fractal checkpoint produced moved_wrong_obj=False on 100% of widowx episodes).
set -uo pipefail
ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD=$ROOT/token-cd
CONDA_ROOT=/home/leju-suzhou/miniconda3
OVERLAY=$ROOT/pi05_sklearn_overlay
TASK="${1:-widowx_carrot_on_plate}"
SEED="${2:-0}"
GPU="${3:-1}"
PORT=8121
OUT=$TOKEN_CD/runs/bridge_check
mkdir -p "$OUT/logs" "$OUT/manifests"
echo "=== bridge checkpoint check: task=$TASK seed=$SEED gpu=$GPU port=$PORT ==="

# manifest with just this one task/seed, vanilla only (fastest signal)
python3 - "$OUT" "$TASK" "$SEED" <<'PY'
import json, sys
out, task, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
json.dump({"worker_id": 0, "jobs": [{"task": task, "seed": seed, "arms": ["vanilla"]}]},
          open(f"{out}/manifests/worker_0.json", "w"), indent=2)
PY

(
  source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate pi05
  export CUDA_VISIBLE_DEVICES="$GPU"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.22
  export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
  export OPENPI_DATA_HOME="$ROOT/checkpoints/openpi-assets" HF_HUB_OFFLINE=1
  export PI0FAST_CD_KMEANS=sklearn PI0FAST_CD_TAP_MODE=separate PI0FAST_CD_DECODE=verified
  export PYTHONPATH="$OVERLAY:$ROOT/openpi/src:$ROOT/openpi/packages/openpi-client/src:$TOKEN_CD"
  exec python -u "$TOKEN_CD/research/semantic_token_cd/pi0_fast_l6_simpler_server.py" \
    --gpu "$GPU" --port "$PORT" --checkpoint "$ROOT/checkpoints/openpi_fast_finetuned/bridge"
) > "$OUT/logs/server_bridge.log" 2>&1 &
SRV=$!
echo "server pid=$SRV; waiting for warmup..."
for i in $(seq 1 120); do
  grep -q "warmups passed" "$OUT/logs/server_bridge.log" 2>/dev/null && { echo "  bridge server READY"; break; }
  kill -0 $SRV 2>/dev/null || { echo "  server DIED"; tail -5 "$OUT/logs/server_bridge.log"; exit 1; }
  sleep 5
done

(
  source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate pi0
  export CUDA_VISIBLE_DEVICES="$GPU"
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
  export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export JAX_PLATFORMS=cpu
  export PYTHONPATH="/home/leju-suzhou/.cache/pi0fast_client_deps:$ROOT/openpi/packages/openpi-client/src:$ROOT/pcd_openvla_simpler_box_31b027e/source:$ROOT/SimplerEnv:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/SpatialVLA/ManiSkill2_real2sim:$TOKEN_CD:$ROOT/openpi/src"
  exec python -u "$TOKEN_CD/research/semantic_token_cd/pi0_fast_simpler_l6_matched_rollout.py" \
    --gpu "$GPU" --worker-id 0 --shard "$OUT/manifests/worker_0.json" \
    --host 127.0.0.1 --port "$PORT" --out "$OUT" --rt-depth 8
) > "$OUT/logs/worker.log" 2>&1
echo "worker exited rc=$?"
kill $SRV 2>/dev/null
echo "=== result ==="
python3 - "$OUT" "$TASK" "$SEED" <<'PY'
import json, sys
out, task, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(f"{out}/episodes/{task}/vanilla_seed{seed:03d}.json"))
fi = d.get("final_info") or {}
print(f"  success={d.get('success')} steps={d.get('steps')} truncated={d.get('truncated')}")
for k in ("is_src_obj_grasped", "moved_correct_obj", "moved_wrong_obj", "src_on_target", "consecutive_grasp"):
    print(f"  {k:22s} = {fi.get(k)}")
print("  VERDICT:", "BRIDGE CHECKPOINT INTERACTS" if (fi.get("moved_wrong_obj") in (True, "True")
      or fi.get("is_src_obj_grasped") in (True, "True")) else
      "still no interaction - investigate further")
PY
