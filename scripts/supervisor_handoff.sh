#!/usr/bin/env bash
# Handoff supervisor: finish the pi0-FAST run, then automatically start the SpatialVLA WidowX
# lambda scan. Everything it does is appended to logs/supervisor_handoff.log and mirrored into
# logs/handoff_status.json so a light patrol only has to read one small file.
#
# Phases
#   WAIT_PI0FAST      poll until the pi0-FAST run reaches 3500/3500 (60 s, max 6 h)
#   SELF_CHECK        scripts/selfcheck_pi0fast_3500.py must exit 0, else STOP with reason
#   STOP_PI0FAST      kill only our pi0_fast_* processes (GPU 4-7 belong to another user)
#   PREFLIGHT         run_spatialvla_widowx_lambda_0_299_tmux.sh preflight, wait for
#                     decoder-parity-PASS, no Traceback
#   FORMAL            run_spatialvla_widowx_lambda_0_299_tmux.sh formal, verify 9 sessions
#   WAIT_WIDOWX_LAMBDA poll the new artifact to 3600/3600 (300 s, max 14 h), then self-check
#
# Safety: it never touches GPU 4-7, never replaces an existing tmux session (the launcher
# refuses), and stops with a FAIL status instead of improvising whenever a gate fails.
set -uo pipefail

T=/home/leju-suzhou/zjt_ws/token-cd
R=$T/runs/pi0fast_l6_matched_9task_100seed
NEW=$T/artifacts/spatialvla_widowx_lambda_0_299_v1
LAUNCH=$T/scripts/run_spatialvla_widowx_lambda_0_299_tmux.sh
SELFCHECK=$T/scripts/selfcheck_pi0fast_3500.py
PY=/home/leju-suzhou/miniconda3/envs/pi05/bin/python
SLOG=$R/logs/supervisor_handoff.log
SJSON=$R/logs/handoff_status.json
PAT_WRK='[p]i0_fast_simpler_l6_matched_rollout'
PAT_SRV='[p]i0_fast_l6_simpler_server'

say() { echo "$(date '+%F %T') $*" >> "$SLOG"; }
status() {  # status <phase> <detail>
  "$PY" - "$1" "$2" "$SJSON" <<'PY'
import json, sys, time, pathlib
phase, detail, path = sys.argv[1], sys.argv[2], sys.argv[3]
p = pathlib.Path(path)
prev = {}
if p.exists():
    try: prev = json.loads(p.read_text())
    except Exception: prev = {}
prev.update({"phase": phase, "detail": detail, "updated": time.strftime("%F %T")})
p.write_text(json.dumps(prev, indent=2, ensure_ascii=False))
PY
  say "STATUS=$1 detail=$2"
}

mkdir -p "$R/logs"
TARGET=3500
if [[ -e "$SJSON" ]]; then
  prev_phase=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('phase',''))" "$SJSON" 2>/dev/null || true)
  if [[ -n "$prev_phase" && "$prev_phase" != "WAIT_PI0FAST" ]]; then
    say "refusing to run twice: previous phase=$prev_phase"
    exit 9
  fi
fi
status WAIT_PI0FAST "waiting for 3500/3500 (target $TARGET)"

# ---------------------------------------------------------------- PHASE 1: wait for 3500 ------
for i in $(seq 1 360); do                     # up to 6 h
  n=$(find "$R/episodes" -name '*.json' 2>/dev/null | wc -l)
  if (( n >= 3500 )); then break; fi
  sleep 60
done
n=$(find "$R/episodes" -name '*.json' 2>/dev/null | wc -l)
if (( n < 3500 )); then
  status TIMEOUT_PI0FAST "only $n/3500 after 6h - not proceeding"
  exit 1
fi
say "pi0-FAST reached $n/3500"

# ---------------------------------------------------------------- PHASE 2: self-check ---------
status SELF_CHECK "running $SELFCHECK"
if ! "$PY" "$SELFCHECK" >> "$SLOG" 2>&1; then
  status FAIL_SELF_CHECK "see logs/handoff_selfcheck.json - manual review required"
  exit 2
fi
status SELF_CHECK_PASS "all gates green"

# ---------------------------------------------------------------- PHASE 3: stop pi0-fast ------
status STOP_PI0FAST "terminating our pi0_fast_* processes only"
pkill -f "$PAT_WRK" 2>/dev/null
sleep 8
pkill -f "$PAT_SRV" 2>/dev/null
sleep 8
for _ in 1 2 3; do
  w=$(pgrep -cf "$PAT_WRK" || true); s=$(pgrep -cf "$PAT_SRV" || true)
  (( w == 0 && s == 0 )) && break
  pkill -9 -f "$PAT_WRK" 2>/dev/null; pkill -9 -f "$PAT_SRV" 2>/dev/null
  sleep 5
done
w=$(pgrep -cf "$PAT_WRK" || true); s=$(pgrep -cf "$PAT_SRV" || true)
if (( w != 0 || s != 0 )); then
  status FAIL_STOP_PI0FAST "workers=$w servers=$s still alive"
  exit 3
fi
say "pi0-fast stopped cleanly; GPUs 1/2/3 free"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 1,2,3 >> "$SLOG"

# ---------------------------------------------------------------- PHASE 4: preflight ----------
status PREFLIGHT "launching preflight (separate artifact)"
if ! bash "$LAUNCH" preflight >> "$SLOG" 2>&1; then
  status FAIL_PREFLIGHT_LAUNCH "launcher refused - see log"
  exit 4
fi
pref_ok=0
for _ in $(seq 1 60); do                      # up to 30 min
  if ! tmux has-session -t svla_wxlam_pre 2>/dev/null; then pref_ok=1; break; fi
  sleep 30
done
PLOG=$T/artifacts/spatialvla_widowx_lambda_0_299_preflight/logs/svla_wxlam_pre.log
if (( pref_ok == 0 )); then
  status FAIL_PREFLIGHT "session still running after 30 min"
  exit 5
fi
if ! grep -q "decoder-parity-PASS" "$PLOG" 2>/dev/null; then
  status FAIL_PREFLIGHT "decoder-parity-PASS not found in $PLOG"
  exit 5
fi
if grep -qiE "traceback|RuntimeError|FileNotFoundError" "$PLOG" 2>/dev/null; then
  status FAIL_PREFLIGHT "error found in $PLOG"
  exit 5
fi
status PREFLIGHT_PASS "decoder-parity-PASS, log clean"

# ---------------------------------------------------------------- PHASE 5: formal -------------
status FORMAL "launching 9 workers (3 per GPU on 1/2/3)"
if ! bash "$LAUNCH" formal >> "$SLOG" 2>&1; then
  status FAIL_FORMAL_LAUNCH "launcher refused - see log"
  exit 6
fi
sleep 20
sess=$(tmux ls 2>/dev/null | grep -c svla_wxlam_g || true)
proc=$(pgrep -cf "^python .*/spatialvla_widowx_lambda_0_299_rollout" || true)
if (( sess != 9 || proc != 9 )); then
  status FAIL_FORMAL_VERIFY "sessions=$sess workers=$proc (expected 9/9)"
  exit 7
fi
n0=$(find "$NEW/episodes" -name 'episode_*_summary.json' 2>/dev/null | wc -l)
status FORMAL_RUNNING "9/9 sessions and workers up; artifact starts at $n0/3600"

# ---------------------------------------------------------------- PHASE 6: wait for 3600 ------
for i in $(seq 1 168); do                     # up to 14 h at 300 s
  n2=$(find "$NEW/episodes" -name 'episode_*_summary.json' 2>/dev/null | wc -l)
  if (( n2 >= 3600 )); then
    status WIDOWX_LAMBDA_COMPLETE "3600/3600 reached"
    exit 0
  fi
  live=$(pgrep -cf "^python .*/spatialvla_widowx_lambda_0_299_rollout" || true)
  if (( live == 0 )); then
    status FAIL_WIDOWX_STALLED "workers exited at $n2/3600 - needs attention"
    exit 8
  fi
  sleep 300
done
n2=$(find "$NEW/episodes" -name 'episode_*_summary.json' 2>/dev/null | wc -l)
status TIMEOUT_WIDOWX "only $n2/3600 after 14 h"
exit 10
