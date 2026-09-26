#!/usr/bin/env bash
set -euo pipefail

WS=/home/leju-suzhou/zjt_ws/token-cd
ROOT="$WS/artifacts/libero_two_bowl_attention_localization_v1_20260926"
SESSION=libero_bowl_attention_localization_20260926
PY=/home/leju-suzhou/zjt_ws/libero-openvla-env/bin/python

mkdir -p "$ROOT/workers" "$ROOT/logs"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 2
fi

mapfile -t JOBS < <("$PY" - <<'PY'
gpus = [1, 1, 1, 2, 2, 2, 3, 3, 3, 6, 6, 6]
cases = [f"{task}:{state}" for task in (2, 8) for state in range(50)]
for worker, gpu in enumerate(gpus):
    shard = ",".join(cases[worker::len(gpus)])
    slot = worker % 3
    print(f"{gpu}\t{slot}\t{shard}")
PY
)

for i in "${!JOBS[@]}"; do
  IFS=$'\t' read -r gpu slot cases <<< "${JOBS[$i]}"
  wid="gpu${gpu}_slot${slot}"
  log="$ROOT/logs/${wid}.log"
  out="$ROOT/workers/${wid}.jsonl"
  command="cd '$WS'; source '$WS/scripts/activate_libero_openvla.sh'; export CUDA_VISIBLE_DEVICES=7,$gpu HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1; python research/semantic_token_cd/libero_two_bowl_localization_audit.py --gpu $gpu --render-gpu 7 --cases '$cases' --output '$out' --worker-id '$wid' > '$log' 2>&1; rc=\$?; echo exit_code=\$rc >> '$log'; exec bash"
  if [[ "$i" == 0 ]]; then
    tmux new-session -d -s "$SESSION" -n "$wid" "bash -lc \"$command\""
  else
    tmux new-window -d -t "$SESSION" -n "$wid" "bash -lc \"$command\""
  fi
done

echo "Started 12 workers in tmux session $SESSION"
tmux list-windows -t "$SESSION"
