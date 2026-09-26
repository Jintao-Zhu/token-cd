#!/usr/bin/env bash
set -u
WS=/home/leju-suzhou/zjt_ws/token-cd
ART="$WS/artifacts/libero_method_component_budget_v1_20260925"
cd "$WS"
mkdir -p "$ART/logs"
start="$(date -Is)"
cat > "$ART/RUN_STATUS.json" <<EOF
{
  "protocol_id": "LIBERO90_L11_MATCHED_FIXED_TASK_M_V1",
  "status": "RUNNING",
  "started_local": "$start",
  "cases": 40,
  "workers": [1, 2, 3, 4, 5, 6],
  "renderer_backend": "osmesa"
}
EOF
pids=()
gpus=(1 2 3 4 5 6)
for gpu in "${gpus[@]}"; do
  research/semantic_token_cd/run_libero_budget_fixed_m_osmesa.sh "$gpu" "fixed_m_gpu${gpu}" "$ART" \
    > "$ART/logs/worker_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
codes=()
for pid in "${pids[@]}"; do
  wait "$pid"
  codes+=("$?")
done
finish="$(date -Is)"
status="COMPLETE"
codes_json=""
for code in "${codes[@]}"; do
  if [[ "$code" != 0 ]]; then status="WORKER_FAILURE"; fi
  if [[ -n "$codes_json" ]]; then codes_json+=", "; fi
  codes_json+="$code"
done
cat > "$ART/RUN_STATUS.json" <<EOF
{
  "protocol_id": "LIBERO90_L11_MATCHED_FIXED_TASK_M_V1",
  "status": "$status",
  "started_local": "$start",
  "finished_local": "$finish",
  "cases": 40,
  "workers": [1, 2, 3, 4, 5, 6],
  "worker_exit_codes": [${codes_json}],
  "renderer_backend": "osmesa"
}
EOF
[[ "$status" == COMPLETE ]]
