#!/usr/bin/env bash
set -u
WS=/home/leju-suzhou/zjt_ws/token-cd
ROOT="$WS/artifacts/prompt_attn_l11_matched_dense_lambda_remaining5_0_99_v1"
WATCH="$WS/research/semantic_token_cd/watch_l11_dense_lambda_remaining5_worker.sh"
RUN="$WS/research/semantic_token_cd/run_l11_dense_lambda_remaining5.sh"
mkdir -p "$ROOT/logs"
pids=()
for slot in g2a g2b g2c g3a g3b g3c; do
  bash "$WATCH" "$slot" >> "$ROOT/logs/$slot.log" 2>&1 &
  pid=$!; printf '%s\n' "$pid" > "$ROOT/logs/$slot.pid"; pids+=("$pid")
done
bash "$RUN" finalize >> "$ROOT/logs/finalize.log" 2>&1 &
pid=$!; printf '%s\n' "$pid" > "$ROOT/logs/finalize.pid"; pids+=("$pid")
printf '[%s] supervisor started workers=%s\n' "$(date --iso-8601=seconds)" "${pids[*]}" | tee -a "$ROOT/logs/supervisor.log"
wait "${pids[@]}"
status=$?
printf '[%s] supervisor exited status=%s\n' "$(date --iso-8601=seconds)" "$status" | tee -a "$ROOT/logs/supervisor.log"
exit "$status"
