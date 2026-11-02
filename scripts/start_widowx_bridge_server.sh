#!/usr/bin/env bash
# Bridge pi0-FAST policy server only (no workers) -- used by the WidowX convention experiments.
#
# Notes learned the hard way:
#   * NO `tee` in the pipeline.  A `pkill` on a pattern that happened to appear in a worker's
#     log path also killed the server's tee, and the server died of SIGPIPE (2026-10-13).
#   * log path deliberately avoids tokens that other cleaners might match.
#   * serves the checkpoint with its OWN norm stats (--norm-stats auto -> bridge_lerobot).
# GPU 1 only (docs/AGENT_CONSTRAINTS.md).
set -uo pipefail

ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD=$ROOT/token-cd
CONDA_ROOT=/home/leju-suzhou/miniconda3
OVERLAY=$ROOT/pi05_sklearn_overlay
CKPT=$ROOT/checkpoints/openpi_fast_finetuned/bridge
SERVER=$TOKEN_CD/research/semantic_token_cd/pi0_fast_l6_simpler_server.py
LOGDIR=$TOKEN_CD/runs/widowx_bridge_servers
GPU="${1:-1}"
PORT="${2:-8121}"
SESS="bridge_server_g${GPU}_p${PORT}"
LOG="$LOGDIR/server_g${GPU}_p${PORT}.log"

case "$GPU" in 1|2|3) ;; *) echo "ABORT: GPU $GPU not authorised (only 1/2/3)" >&2; exit 2;; esac
mkdir -p "$LOGDIR"

if ss -ltn 2>/dev/null | grep -q ":$PORT "; then echo "already listening on $PORT"; exit 0; fi
if tmux has-session -t "$SESS" 2>/dev/null; then echo "ABORT: tmux session $SESS exists" >&2; exit 1; fi

tmux new-session -d -s "$SESS" "bash -lc 'source $CONDA_ROOT/etc/profile.d/conda.sh; conda activate pi05; export CUDA_VISIBLE_DEVICES=$GPU; export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.18; export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1; export OPENPI_DATA_HOME=$ROOT/checkpoints/openpi-assets HF_HUB_OFFLINE=1; export PI0FAST_CD_KMEANS=sklearn PI0FAST_CD_TAP_MODE=separate PI0FAST_CD_DECODE=verified; export PYTHONPATH=$OVERLAY:$ROOT/openpi/src:$ROOT/openpi/packages/openpi-client/src:$TOKEN_CD; python -u $SERVER --gpu $GPU --port $PORT --checkpoint $CKPT --norm-stats auto >> $LOG 2>&1'"

echo "starting $SESS on GPU$GPU port$PORT; waiting for warmup..."
for i in $(seq 1 60); do
  grep -q "warmups passed" "$LOG" 2>/dev/null && { echo "server READY ($(grep -m1 'with norm stats' "$LOG" | tail -c 90))"; exit 0; }
  sleep 5
done
echo "TIMEOUT waiting for warmup; tail:"; tail -5 "$LOG"; exit 1
