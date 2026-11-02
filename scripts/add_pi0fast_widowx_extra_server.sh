#!/usr/bin/env bash
# Add an EXTRA pi0-FAST policy server on one GPU and re-point some workers at it.
#
# Why: the openpi websocket server calls policy.infer() synchronously inside its asyncio
# handler, so one server == one inference at a time.  With 6 servers the whole run is capped
# at ~144 eps/h no matter how many workers are added; a 7th/8th/9th server raises the
# inference concurrency and (if the GPU has spare capacity) the throughput.
#
# GPU policy: only GPU 1/2/3 (docs/AGENT_CONSTRAINTS.md). Never GPU 0, never 4-7.
#
# Usage:
#   bash add_pi0fast_widowx_extra_server.sh --gpu 1 --port 8121 [--move 0,1] [--dry-run]
#     --move  comma-separated worker indices (0..17, i.e. pf_wx18_wNN / manifests18/worker_NN.json)
#             to restart against the new port.  Their shards are unchanged, so no work moves.
set -uo pipefail

GPU=""; PORT=""; MOVE=""; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --move) MOVE="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
[[ -z "$GPU" || -z "$PORT" ]] && { echo "need --gpu and --port" >&2; exit 2; }

ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD=$ROOT/token-cd
CONDA_ROOT=/home/leju-suzhou/miniconda3
OVERLAY=$ROOT/pi05_sklearn_overlay
RUN=$TOKEN_CD/runs/pi0fast_bridge_widowx_4task_100seed
CKPT=$ROOT/checkpoints/openpi_fast_finetuned/bridge
SERVER=$TOKEN_CD/research/semantic_token_cd/pi0_fast_l6_simpler_server.py
ROLLOUT=$TOKEN_CD/research/semantic_token_cd/pi0_fast_simpler_l6_matched_rollout.py
LOGDIR=$RUN/logs/w18
mkdir -p "$LOGDIR"

SESS=$(tmux ls 2>/dev/null | cut -d: -f1 | grep -E "^pf_wx_srv_g${GPU}[a-z]$" | tail -1)
if [[ -z "$SESS" ]]; then echo "no existing server session on GPU$GPU" >&2; exit 2; fi
TAG=$(echo "$SESS" | tail -c 2)
NEXT_TAG=$(printf "\\$(printf '%03o' $(( $(printf '%d' "'$TAG") + 1 )))")
NEWSESS="pf_wx_srv_g${GPU}${NEXT_TAG}"

echo "GPU$GPU: existing servers: $(tmux ls 2>/dev/null | cut -d: -f1 | grep -c "^pf_wx_srv_g${GPU}") ; new session $NEWSESS port $PORT"
if [[ "$DRY" == 1 ]]; then echo "(dry run)"; exit 0; fi

if tmux has-session -t "$NEWSESS" 2>/dev/null; then
  echo "$NEWSESS already up"
else
  tmux new-session -d -s "$NEWSESS" "bash -lc 'source $CONDA_ROOT/etc/profile.d/conda.sh; conda activate pi05; export CUDA_VISIBLE_DEVICES=$GPU; export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.18; export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1; export OPENPI_DATA_HOME=$ROOT/checkpoints/openpi-assets HF_HUB_OFFLINE=1; export PI0FAST_CD_KMEANS=sklearn PI0FAST_CD_TAP_MODE=separate PI0FAST_CD_DECODE=verified; export PYTHONPATH=$OVERLAY:$ROOT/openpi/src:$ROOT/openpi/packages/openpi-client/src:$TOKEN_CD; python -u $SERVER --gpu $GPU --port $PORT --checkpoint $CKPT --norm-stats auto >> $RUN/logs/server_g${GPU}_p${PORT}.log 2>&1'"
  echo "started $NEWSESS (gpu$GPU port$PORT)"
fi

echo "waiting for warmup on port $PORT ..."
ok=0
for i in $(seq 1 90); do
  grep -q "warmups passed" "$RUN/logs/server_g${GPU}_p${PORT}.log" 2>/dev/null && { ok=1; break; }
  sleep 5
done
[[ "$ok" == 1 ]] && echo "port$PORT READY: $(grep -m1 'with norm stats' "$RUN/logs/server_g${GPU}_p${PORT}.log" | tail -c 70)" \
                || { echo "port$PORT FAILED to warm up"; tail -5 "$RUN/logs/server_g${GPU}_p${PORT}.log"; exit 3; }

if [[ -n "$MOVE" ]]; then
  IFS=',' read -ra IDS <<< "$MOVE"
  for i in "${IDS[@]}"; do
    sess="pf_wx18_w$(printf '%02d' "$i")"
    log="$LOGDIR/worker_$(printf '%02d' "$i")_gpu${GPU}_p${PORT}.log"
    tmux kill-session -t "$sess" 2>/dev/null && echo "killed $sess" || echo "$sess not running"
    tmux new-session -d -s "$sess" "bash -lc 'source $CONDA_ROOT/etc/profile.d/conda.sh; conda activate pi0; export CUDA_VISIBLE_DEVICES=$GPU; export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json; export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; export JAX_PLATFORMS=cpu; export PI0FAST_WIDOWX_LAYOUT=bridge; export PYTHONPATH=/home/leju-suzhou/.cache/pi0fast_client_deps:$ROOT/openpi/packages/openpi-client/src:$ROOT/pcd_openvla_simpler_box_31b027e/source:$ROOT/SimplerEnv:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/SpatialVLA/ManiSkill2_real2sim:$TOKEN_CD:$ROOT/openpi/src; python -u $ROLLOUT --gpu $GPU --worker-id $i --shard $RUN/manifests18/worker_$i.json --host 127.0.0.1 --port $PORT --out $RUN --checkpoint $CKPT --rt-depth 8 >> $log 2>&1'"
    echo "restarted $sess -> port$PORT (shard $i unchanged)"
    sleep 1
  done
fi

echo "--- state ---"
tmux ls 2>/dev/null | cut -d: -f1 | grep -E 'pf_wx' | sort | tr '\n' ' '; echo
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed -n '2,4p'
