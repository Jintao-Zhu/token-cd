#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD="$ROOT/token-cd"
PYTHON="$ROOT/openvla-ar-h100/bin/python"
RUNNER="$TOKEN_CD/research/semantic_token_cd/spatialvla_three_studies_rollout.py"
SNAPSHOTS="$TOKEN_CD/artifacts/vanilla_recon_shr_canonical_0_299_v2"
ARTIFACT="$TOKEN_CD/artifacts/spatialvla_l10_vanilla_expand9_to300_v1"
export PYTHONPATH="$ROOT/spatialvla-runtime-overlay-20260928:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/pcd_openvla_simpler_box_31b027e/source:$TOKEN_CD/task1/shim_site:$TOKEN_CD"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

cd "$TOKEN_CD"
[[ -x "$PYTHON" && -f "$RUNNER" && -d "$SNAPSHOTS" && -f "$ARTIFACT/CONFIG_LOCK.json" ]] || { echo "runner, snapshots, or existing expansion lock missing" >&2; exit 2; }

declare -A BODY
add() { local key=$1 tasks=$2 seeds=$3 setup=$4 label=$5; BODY[$key]+="run_job '$tasks' '$seeds' '$setup' '$label'\n"; }

add g1w1 google_robot_open_drawer 74-99 google_robot open_tail
add g1w1 google_robot_close_drawer 297-299 google_robot close_tail
add g1w2 google_robot_pick_coke_can 200-233 google_robot pick_a
add g1w3 google_robot_pick_coke_can 234-266 google_robot pick_b
add g1w4 google_robot_place_apple_in_closed_top_drawer 47-64 google_robot apple_a
add g1w5 google_robot_place_apple_in_closed_top_drawer 65-82 google_robot apple_b
add g2w1 google_robot_pick_coke_can 267-299 google_robot pick_c
add g2w2 google_robot_place_apple_in_closed_top_drawer 83-99 google_robot apple_c
add g2w3 google_robot_place_apple_in_closed_top_drawer 250-274 google_robot apple_d
add g2w4 google_robot_place_apple_in_closed_top_drawer 275-299 google_robot apple_e
add g3w1 widowx_carrot_on_plate 200-224 widowx_bridge carrot_a
add g3w2 widowx_carrot_on_plate 225-249 widowx_bridge carrot_b
add g3w3 widowx_carrot_on_plate 250-274 widowx_bridge carrot_c
add g3w4 widowx_carrot_on_plate 275-299 widowx_bridge carrot_d

for gpu in 1 2 3; do
  for slot in 1 2 3 4 5; do
    key="g${gpu}w${slot}"
    [[ -n "${BODY[$key]:-}" ]] || continue
    session="svla4700r_${key}"
    tmux has-session -t "$session" 2>/dev/null && { echo "tmux session already exists: $session" >&2; exit 2; }
  done
done

for gpu in 1 2 3; do
  for slot in 1 2 3 4 5; do
    key="g${gpu}w${slot}"
    [[ -n "${BODY[$key]:-}" ]] || continue
    session="svla4700r_${key}"
    worker="$ARTIFACT/logs/${session}.sh"
    launcher_log="$ARTIFACT/logs/${session}.launcher.log"
    mkdir -p "$(dirname "$worker")"
    {
      echo '#!/usr/bin/env bash'
      echo 'set -euo pipefail'
      printf 'export PYTHONPATH=%q HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json\n' "$PYTHONPATH"
      printf 'PYTHON=%q RUNNER=%q SNAPSHOTS=%q ARTIFACT=%q GPU=%q WORKER=%q\n' "$PYTHON" "$RUNNER" "$SNAPSHOTS" "$ARTIFACT" "$gpu" "$key"
      cat <<'SH'
run_job() {
  local tasks=$1 seeds=$2 setup=$3 label=$4
  local log="$ARTIFACT/logs/${WORKER}_${label}.log"
  echo "[$(date -Is)] START tasks=$tasks seeds=$seeds setup=$setup GPU=$GPU"
  "$PYTHON" -u "$RUNNER" --study expand9_to300 --artifact "$ARTIFACT" \
    --snapshot-artifact "$SNAPSHOTS" --gpu "$GPU" --worker-id "$WORKER" \
    --tasks "$tasks" --seeds "$seeds" --model-setup "$setup" >"$log" 2>&1
  echo "[$(date -Is)] DONE tasks=$tasks seeds=$seeds log=$log"
}
SH
      printf '%b' "${BODY[$key]}"
      echo "echo 'worker complete: $session'"
    } > "$worker"
    chmod +x "$worker"
    tmux new-session -d -s "$session" "bash '$worker' > '$launcher_log' 2>&1"
    echo "started $session GPU=$gpu"
  done
done
echo "resumed missing seeds only; active slots GPU1=5 GPU2=4 GPU3=4; outstanding episode records at repartition=661"
