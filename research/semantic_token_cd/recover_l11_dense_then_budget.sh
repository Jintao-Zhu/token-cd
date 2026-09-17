#!/usr/bin/env bash
set -euo pipefail

WS=/home/leju-suzhou/zjt_ws/token-cd
DENSE="$WS/artifacts/prompt_attn_l11_matched_dense_lambda_0_99_v1"
POLICY="$WS/research/semantic_token_cd/prompt_attn_shr_policy.py"
DENSE_POLICY="$DENSE/recovery/prompt_attn_shr_policy_dense_locked.py"
BUDGET_POLICY="$DENSE/recovery/prompt_attn_shr_policy_budget_version.py"
LOG="$DENSE/logs/recovery_supervisor.log"
EXPECTED_DENSE_SHA=c09582d1e8ff7210434bfcb0e3954ce39af82efe911034ebe92b63686c5647ee
EXPECTED_BUDGET_SHA=5620ed356a6ff0070125a58b8ff6a447e81f001dbb31b384da593c2f78cdddac

sha() {
  sha256sum "$1" | awk '{print $1}'
}

restore_budget_policy() {
  if [[ -f "$BUDGET_POLICY" ]]; then
    cp "$BUDGET_POLICY" "$POLICY.tmp"
    mv "$POLICY.tmp" "$POLICY"
  fi
}

trap restore_budget_policy EXIT

if [[ "$(sha "$DENSE_POLICY")" != "$EXPECTED_DENSE_SHA" ]]; then
  echo "dense policy recovery hash mismatch" >&2
  exit 1
fi
if [[ "$(sha "$BUDGET_POLICY")" != "$EXPECTED_BUDGET_SHA" ]]; then
  echo "budget policy backup hash mismatch" >&2
  exit 1
fi

cp "$DENSE_POLICY" "$POLICY.tmp"
mv "$POLICY.tmp" "$POLICY"
if [[ "$(sha "$POLICY")" != "$EXPECTED_DENSE_SHA" ]]; then
  echo "failed to install locked dense policy" >&2
  exit 1
fi

echo "[$(date --iso-8601=seconds)] resuming dense sweep from existing summaries" | tee -a "$LOG"
bash "$WS/research/semantic_token_cd/supervise_l11_dense_lambda3200.sh" >> "$LOG" 2>&1
if [[ ! -f "$DENSE/COMPLETE.json" ]]; then
  echo "dense supervisor exited without COMPLETE.json" >&2
  exit 1
fi

restore_budget_policy
trap - EXIT
if [[ "$(sha "$POLICY")" != "$EXPECTED_BUDGET_SHA" ]]; then
  echo "failed to restore budget policy" >&2
  exit 1
fi

echo "[$(date --iso-8601=seconds)] dense complete; starting budget causal pilot" | tee -a "$LOG"
exec bash "$WS/research/semantic_token_cd/run_l11_budget_causal_pilot.sh"
