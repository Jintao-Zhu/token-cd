#!/usr/bin/env bash
# Launch the frozen pi0-FAST L6-Matched 9-task experiment: 3 policy servers (one per GPU on
# 1/2/3) plus 9 rollout workers (3 per GPU, one task each, 500 episodes each = 4500).
#
# Usage:
#   launch_9workers.sh            # full run
#   launch_9workers.sh --smoke    # seed 0 only, all five arms (45 episodes)
#   launch_9workers.sh --preflight-only
#
# Design notes:
#   * Servers hold the model; workers hold the SIMPLER/SAPIEN environment. Workers connect
#     over websocket, so a worker crash never takes the model down with it.
#   * Everything is resumable: the worker skips any episode whose result file is already
#     status=complete, so re-running this script continues rather than duplicating.
#   * One (task, seed, arm) maps to exactly one result path, so no episode can be counted
#     twice even if two workers are mistakenly pointed at the same shard.
#   * Nothing here sets PI0FAST_CD_EMBED_JIT or PI0FAST_CD_BATCH to 1: both were rejected
#     for breaking token equivalence.
set -uo pipefail

ROOT="/home/leju-suzhou/zjt_ws"
TOKEN_CD="$ROOT/token-cd"
CONDA_ROOT="/home/leju-suzhou/miniconda3"
OVERLAY="$ROOT/pi05_sklearn_overlay"
RUN_DIR="$TOKEN_CD/runs/pi0fast_l6_matched_9task_100seed"
MANIFEST_DIR="$RUN_DIR/manifests"
LOG_DIR="$RUN_DIR/logs"
# Checkpoints per task family. fractal is the google_robot fine-tune, bridge is the widowx
# fine-tune. Using fractal for widowx tasks made the robot never touch any object
# (moved_wrong_obj=False on 100% of 163 episodes): the action normalisation statistics differ,
# so the policy output was meaningless in that embodiment.
CKPT_GOOGLE="$ROOT/checkpoints/openpi_fast_finetuned/fractal"
CKPT_WIDOWX="$ROOT/checkpoints/openpi_fast_finetuned/bridge"

task_checkpoint() {
  case "$1" in
    google_robot_*) echo "$CKPT_GOOGLE" ;;
    widowx_*)       echo "$CKPT_WIDOWX" ;;
    *)              echo "$CKPT_GOOGLE" ;;
  esac
}

task_family() {
  case "$1" in
    google_robot_*) echo "google" ;;
    widowx_*)       echo "widowx" ;;
    *)              echo "google" ;;
  esac
}
ROLLOUT="$TOKEN_CD/research/semantic_token_cd/pi0_fast_simpler_l6_matched_rollout.py"
SERVER="$TOKEN_CD/research/semantic_token_cd/pi0_fast_l6_simpler_server.py"

GPUS=(1 1 1 1 1 2 2 2 2 2 3 3 3 3 3)
# Five google_robot tasks, each split into three seed chunks, giving 15 workers.
# build_manifest emits shards in the order task0-chunk0..2, task1-chunk0..2, ... so this list
# must follow the same order for worker w to receive the shard it is supposed to run.
TASKS=(
  google_robot_open_drawer
  google_robot_open_drawer
  google_robot_open_drawer
  google_robot_close_drawer
  google_robot_close_drawer
  google_robot_close_drawer
  google_robot_pick_coke_can
  google_robot_pick_coke_can
  google_robot_pick_coke_can
  google_robot_move_near
  google_robot_move_near
  google_robot_move_near
  google_robot_place_apple_in_closed_top_drawer
  google_robot_place_apple_in_closed_top_drawer
  google_robot_place_apple_in_closed_top_drawer
)
# Servers per GPU. The per-server guided decode is dispatch-bound (one server uses ~1 core
# while the GPU sits at ~22%), and measured capacity is 5.7 replans/min per server against a
# much higher demand, so a second independent process nearly doubles throughput. Memory per
# GPU: SERVERS_PER_GPU * 17 GB (model) + 3 * 3.2 GB (renderers); 2 fits comfortably in 80 GB.
SERVERS_PER_GPU="${PI0FAST_SERVERS_PER_GPU:-2}"
SERVER_PORTS=(8101 8102 8103 8111 8112 8113)
SERVER_GPUS=(1 2 3 1 2 3)
# RT-full rendering as frozen in config.yaml.
RT_DEPTH="${PI0FAST_RT_DEPTH:-8}"

MODE="full"
case "${1:-}" in
  --smoke) MODE="smoke" ;;
  --preflight-only) MODE="preflight" ;;
  "") ;;
  *) echo "unknown option $1" >&2; exit 2 ;;
esac

mkdir -p "$LOG_DIR" "$MANIFEST_DIR" "$RUN_DIR/episodes"
cd "$TOKEN_CD" || exit 1

# ---------------------------------------------------------------------------
# Frozen-config guard: refuse to start if the runtime environment disagrees.
# ---------------------------------------------------------------------------
for v in PI0FAST_CD_KMEANS PI0FAST_CD_TAP_MODE PI0FAST_CD_DECODE PI0FAST_CD_EMBED_JIT PI0FAST_CD_BATCH; do
  if [[ -n "${!v:-}" ]]; then
    echo "refusing to start: $v is set in the environment (${!v}); the frozen configuration" >&2
    echo "must be supplied by this script, not inherited." >&2
    exit 3
  fi
done

echo "=== mode=$MODE rt_depth=$RT_DEPTH ==="
echo "=== generating manifests ==="
if [[ "$MODE" == "smoke" ]]; then
  MANIFEST_SUBDIR="manifests_smoke"
  SEED_ARGS=(--seed-start 0 --seed-end 0)
else
  MANIFEST_SUBDIR="manifests"
  SEED_ARGS=(--seed-start 0 --seed-end 99)
fi
PYTHONPATH="$ROOT/openpi/packages/openpi-client/src:$ROOT/openpi/src:$TOKEN_CD" \
  "$CONDA_ROOT/envs/pi05/bin/python" "$ROLLOUT" --gpu 0 --manifest-only \
  --manifest-subdir "$MANIFEST_SUBDIR" "${SEED_ARGS[@]}" --out "$RUN_DIR" \
  || { echo "manifest generation failed" >&2; exit 4; }
MANIFEST_DIR="$RUN_DIR/$MANIFEST_SUBDIR"

CONFIG_HASH=$(sha256sum "$RUN_DIR/config.yaml" | awk '{print $1}')
PROTO_HASH=$(sha256sum "$RUN_DIR/protocol.json" | awk '{print $1}')
echo "config.yaml sha256=$CONFIG_HASH"
echo "protocol.json sha256=$PROTO_HASH"
{
  echo "config_sha256=$CONFIG_HASH"
  echo "protocol_sha256=$PROTO_HASH"
  echo "git_token_cd=$(git -C "$TOKEN_CD" rev-parse HEAD 2>/dev/null)"
  echo "git_openpi=$(git -C "$ROOT/openpi" rev-parse HEAD 2>/dev/null)"
  echo "checkpoint_google=$CKPT_GOOGLE"
  echo "checkpoint_widowx=$CKPT_WIDOWX"
  echo "rt_depth=$RT_DEPTH"
  echo "mode=$MODE"
  echo "started=$(date '+%F %T')"
} > "$RUN_DIR/frozen_manifest.txt"
cat "$RUN_DIR/frozen_manifest.txt"

if [[ "$MODE" == "preflight" ]]; then
  echo "=== preflight only; not starting servers or workers ==="
  exit 0
fi

# ---------------------------------------------------------------------------
# Policy servers: SERVERS_PER_GPU per physical GPU, each holding its own model copy.
# ---------------------------------------------------------------------------
SERVER_PIDS=()
SERVER_PORT_LIST=()
SERVER_LOG_LIST=()
for i in "${!SERVER_GPUS[@]}"; do
  gpu="${SERVER_GPUS[$i]}"; port="${SERVER_PORTS[$i]}"
  # only launch the first SERVERS_PER_GPU entries of each GPU group
  nth=0
  for j in $(seq 0 $((i - 1))); do
    [ "${SERVER_GPUS[$j]}" = "$gpu" ] && nth=$((nth + 1))
  done
  if [ "$nth" -ge "$SERVERS_PER_GPU" ]; then continue; fi
  # server slot 0 on a GPU serves google_robot tasks, slot 1 serves widowx tasks
  if [ "$nth" -eq 0 ]; then
    server_ckpt="$CKPT_GOOGLE"; server_family="google-a"
  else
    server_ckpt="$CKPT_GOOGLE"; server_family="google-b"
  fi
  log="$LOG_DIR/server_gpu${gpu}_port${port}_${server_family}.log"
  echo "starting server gpu=$gpu port=$port family=$server_family ckpt=$(basename "$server_ckpt") -> $log"
  (
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate pi05
    export CUDA_VISIBLE_DEVICES="$gpu"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    # Split the card between the model copies; measured model footprint is ~17 GB.
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${PI0FAST_SERVER_MEM_FRACTION:-0.22}"
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    export OPENPI_DATA_HOME="$ROOT/checkpoints/openpi-assets"
    export HF_HUB_OFFLINE=1
    # Frozen configuration, set here and nowhere else.
    export PI0FAST_CD_KMEANS=sklearn
    export PI0FAST_CD_TAP_MODE=separate
    export PI0FAST_CD_DECODE=verified
    export PI0FAST_CD_EMBED_JIT=0
    export PI0FAST_CD_BATCH=0
    export PI0FAST_TIMING=0
    export PYTHONPATH="$OVERLAY:$ROOT/openpi/src:$ROOT/openpi/packages/openpi-client/src:$TOKEN_CD"
    exec python -u "$SERVER" --gpu "$gpu" --port "$port" --checkpoint "$server_ckpt"
  ) > "$log" 2>&1 &
  SERVER_PIDS+=("$!")
  SERVER_PORT_LIST+=("$port")
  SERVER_LOG_LIST+=("$log")
  sleep 5
done

# Wait for every launched server to report that both arms warmed up.
echo "=== waiting for ${#SERVER_PIDS[@]} servers to warm up (this compiles both arms) ==="
for i in "${!SERVER_PIDS[@]}"; do
  log="${SERVER_LOG_LIST[$i]}"; port="${SERVER_PORT_LIST[$i]}"
  for attempt in $(seq 1 240); do
    if grep -q "Vanilla and L6-Matched warmups passed" "$log" 2>/dev/null; then
      echo "  port$port ready"
      break
    fi
    if ! kill -0 "${SERVER_PIDS[$i]}" 2>/dev/null; then
      echo "  port$port server DIED; see $log" >&2
      exit 5
    fi
    sleep 5
  done
  if ! grep -q "Vanilla and L6-Matched warmups passed" "$log" 2>/dev/null; then
    echo "  port$port did not become ready in time; see $log" >&2
    exit 6
  fi
done

# ---------------------------------------------------------------------------
# Rollout workers: 9 of them, 3 per GPU, one task each.
# ---------------------------------------------------------------------------
WORKER_PIDS=()
for w in "${!TASKS[@]}"; do
  gpu="${GPUS[$w]}"
  task="${TASKS[$w]}"
  # All workers now run google_robot tasks, so route them across this GPU's servers
  # round-robin to keep every server process busy.
  ports_of_gpu=()
  for j in "${!SERVER_GPUS[@]}"; do
    [ "${SERVER_GPUS[$j]}" = "$gpu" ] || continue
    nth=0
    for k in $(seq 0 $((j - 1))); do
      [ "${SERVER_GPUS[$k]}" = "$gpu" ] && nth=$((nth + 1))
    done
    [ "$nth" -ge "$SERVERS_PER_GPU" ] && continue
    ports_of_gpu+=("${SERVER_PORTS[$j]}")
  done
  idx=0
  for j in $(seq 0 $((w - 1))); do
    [ "${GPUS[$j]}" = "$gpu" ] && idx=$((idx + 1))
  done
  port="${ports_of_gpu[$(( idx % ${#ports_of_gpu[@]} ))]}"
  shard="$MANIFEST_DIR/worker_${w}.json"
  log="$LOG_DIR/worker_${w}_gpu${gpu}_${task}.log"
  if [[ ! -f "$shard" ]]; then
    echo "missing shard $shard" >&2; exit 7
  fi
  echo "starting worker $w gpu=$gpu port=$port task=$task -> $log"
  (
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate pi0
    export CUDA_VISIBLE_DEVICES="$gpu"
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.10
    export JAX_PLATFORMS=cpu
    export PYTHONPATH="/home/leju-suzhou/.cache/pi0fast_client_deps:$ROOT/openpi/packages/openpi-client/src:$ROOT/pcd_openvla_simpler_box_31b027e/source:$ROOT/SimplerEnv:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/SpatialVLA/ManiSkill2_real2sim:$TOKEN_CD:$ROOT/openpi/src"
    exec python -u "$ROLLOUT" \
      --gpu "$gpu" --worker-id "$w" --shard "$shard" \
      --host 127.0.0.1 --port "$port" \
      --out "$RUN_DIR" --checkpoint "$(task_checkpoint "$task")" --rt-depth "$RT_DEPTH"
  ) > "$log" 2>&1 &
  WORKER_PIDS+=("$!")
  sleep 2
done

echo "=== all $((${#SERVER_PIDS[@]} + ${#WORKER_PIDS[@]})) processes started ==="
echo "servers: ${SERVER_PIDS[*]}"
echo "workers: ${WORKER_PIDS[*]}"

STATUS=0
for pid in "${WORKER_PIDS[@]}"; do
  wait "$pid" || STATUS=1
done
echo "=== all workers finished, status=$STATUS ==="
# Servers are deliberately LEFT RUNNING. Killing them here meant that stopping a worker by
# hand (to rebalance the tail) also tore down the model servers, because this cleanup fires
# as soon as the wait loop drains. Shut them down explicitly when the whole run is done.
echo "servers left running (pids: ${SERVER_PIDS[*]}); stop them explicitly if needed"
exit "$STATUS"
