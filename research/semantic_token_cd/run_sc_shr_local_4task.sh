#!/usr/bin/env bash
# SC-SHR single-arm local paired rollout: 4 tasks x seeds 300-399.
# References: artifacts/spatial_harmonic_recon_v1 (vanilla/SHR baselines).
set -u

REPO=/data/docker/dev_zjt/data/code
cd "$REPO"

PY=$REPO/task1/.venvs/openvla-ar/bin/python
SCD=research/semantic_token_cd
ART=$REPO/artifacts/sc_shr_local_v1
SRC=$REPO/artifacts/spatial_harmonic_recon_v1
PCD_SOURCE=$REPO/official-reproductions/pcd_openvla_simpler_box_31b027e/source/PCD

export TOKEN_CD_REPO_ROOT="$REPO"
export TOKEN_CD_PCD_ROOT="$REPO/official-reproductions/pcd_openvla_simpler_box_31b027e"
export TOKEN_CD_PCD_SOURCE="$PCD_SOURCE"
export PYTHONPATH="$REPO/task1/shim_site:$REPO:$PCD_SOURCE"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

GPU=${GPU:-0}
mkdir -p "$ART/logs"

TASKS=(
  google_robot_close_drawer
  google_robot_open_drawer
  google_robot_pick_coke_can
  google_robot_move_near
)

pids=()
for task in "${TASKS[@]}"; do
  short=$(echo "$task" | sed 's/google_robot_//')
  log="$ART/logs/${short}.log"
  if [ -f "$ART/logs/${short}.done" ]; then
    echo "[sc_shr] skip done task=$task"
    continue
  fi
  echo "[sc_shr] starting task=$task gpu=$GPU"
  (
    CUDA_VISIBLE_DEVICES=$GPU "$PY" \
      "$SCD/sc_shr_local_rollout.py" --task "$task" --seeds 300-399 \
      --artifact "$ART" --source-artifact "$SRC" --gpu "$GPU" \
      --worker-id "local-${short}" > "$log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then touch "$ART/logs/${short}.done"; fi
    echo "task=$task rc=$rc" >> "$ART/logs/${short}.log"
  ) &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "SC-SHR local rollout loop finished."
echo "summaries=$(find "$ART/episodes" -name 'episode_*_summary.json' 2>/dev/null | wc -l)"
