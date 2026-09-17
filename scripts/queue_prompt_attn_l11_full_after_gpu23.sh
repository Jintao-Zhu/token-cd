#!/usr/bin/env bash
set -euo pipefail

repo=/home/leju-suzhou/zjt_ws/token-cd
cd "$repo"
mkdir -p logs

lock=logs/queue_prompt_attn_l11_full_after_gpu23.lock
exec 9>"$lock"
if ! flock -n 9; then
  exit 0
fi

log=logs/queue_prompt_attn_l11_full_after_gpu23.log
exec >>"$log" 2>&1
echo "$(date -Is) waiting for target_boost_confidence_gate_v1 to finish"
echo "$(date -Is) queue pid=$$"

current_complete=artifacts/target_positive_boost_confidence_gate_v1/COMPLETE

while true; do
  target_done=0
  if [[ -f "$current_complete" ]] && ! pgrep -f 'target_boost_confidence_(coordinator|rollout)\.py' >/dev/null 2>&1; then
    target_done=1
  fi

  gpu23_free=1
  if ! python3 - <<'PY'
import subprocess
mapping = {}
for line in subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
    text=True,
).splitlines():
    index, uuid = [x.strip() for x in line.split(",", 1)]
    mapping[uuid] = int(index)
apps = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"],
    text=True,
).splitlines()
busy = sorted({mapping.get(line.strip(), -1) for line in apps if line.strip()})
raise SystemExit(0 if not ({2, 3} & set(busy)) else 1)
PY
  then
    gpu23_free=0
  fi

  if [[ "$target_done" -eq 1 && "$gpu23_free" -eq 1 ]]; then
    break
  fi
  sleep 30
done

echo "$(date -Is) GPU2/3 free; starting L11-Matched canonical seeds 100-299"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$repo:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source:${PYTHONPATH:-}"
exec /home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python \
  research/semantic_token_cd/launch_prompt_attn_l11_full_0_299.py
