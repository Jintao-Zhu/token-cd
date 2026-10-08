#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD="$ROOT/token-cd"
PYTHON="$ROOT/openvla-ar-h100/bin/python"
RUNNER="$TOKEN_CD/research/semantic_token_cd/spatialvla_three_studies_rollout.py"
SNAPSHOTS="$TOKEN_CD/artifacts/vanilla_recon_shr_canonical_0_299_v2"
ART_DRAWERS="$TOKEN_CD/artifacts/spatialvla_drawers_rt_depth0_vanilla_l10_100_199_v1"
ART_RTFULL="$TOKEN_CD/artifacts/spatialvla_remaining7_rtfull_vanilla_100_199_v1"
ART_EXPAND="$TOKEN_CD/artifacts/spatialvla_l10_vanilla_expand9_to300_v1"
export PYTHONPATH="$ROOT/spatialvla-runtime-overlay-20260928:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/pcd_openvla_simpler_box_31b027e/source:$TOKEN_CD/task1/shim_site:$TOKEN_CD"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

cd "$TOKEN_CD"
[[ -x "$PYTHON" && -f "$RUNNER" && -d "$SNAPSHOTS" ]] || { echo "Python, runner, or canonical snapshots missing" >&2; exit 2; }

declare -A BODY
add() { local key=$1 study=$2 artifact=$3 tasks=$4 seeds=$5 setup=$6; BODY[$key]+="run_job '$study' '$artifact' '$tasks' '$seeds' '$setup'\n"; }

add g1w1 drawers_rt_depth0 "$ART_DRAWERS" google_robot_open_drawer 100-199 google_robot
add g1w1 expand9_to300 "$ART_EXPAND" google_robot_open_drawer 0-99 google_robot
add g1w2 drawers_rt_depth0 "$ART_DRAWERS" google_robot_close_drawer 100-199 google_robot
add g1w2 expand9_to300 "$ART_EXPAND" google_robot_close_drawer 0-99 google_robot
add g1w3 remaining7_rtfull_vanilla "$ART_RTFULL" google_robot_pick_coke_can 100-199 google_robot
add g1w3 expand9_to300 "$ART_EXPAND" google_robot_pick_coke_can 0-99 google_robot
add g1w4 remaining7_rtfull_vanilla "$ART_RTFULL" google_robot_move_near 100-199 google_robot
add g1w4 expand9_to300 "$ART_EXPAND" google_robot_move_near 0-99 google_robot
add g2w1 remaining7_rtfull_vanilla "$ART_RTFULL" google_robot_place_apple_in_closed_top_drawer 100-199 google_robot
add g2w1 expand9_to300 "$ART_EXPAND" google_robot_place_apple_in_closed_top_drawer 0-99 google_robot
add g2w2 remaining7_rtfull_vanilla "$ART_RTFULL" widowx_carrot_on_plate 100-199 widowx_bridge
add g2w2 expand9_to300 "$ART_EXPAND" widowx_carrot_on_plate 0-99 widowx_bridge
add g2w3 remaining7_rtfull_vanilla "$ART_RTFULL" widowx_put_eggplant_in_basket 100-199 widowx_bridge
add g2w3 expand9_to300 "$ART_EXPAND" widowx_put_eggplant_in_basket 0-99 widowx_bridge
add g2w4 remaining7_rtfull_vanilla "$ART_RTFULL" widowx_spoon_on_towel 100-199 widowx_bridge
add g2w4 expand9_to300 "$ART_EXPAND" widowx_spoon_on_towel 0-99 widowx_bridge
add g3w1 remaining7_rtfull_vanilla "$ART_RTFULL" widowx_stack_cube 100-199 widowx_bridge
add g3w1 expand9_to300 "$ART_EXPAND" widowx_stack_cube 0-99 widowx_bridge
add g3w2 expand9_to300 "$ART_EXPAND" google_robot_open_drawer,google_robot_close_drawer,google_robot_pick_coke_can 200-299 google_robot
add g3w3 expand9_to300 "$ART_EXPAND" google_robot_move_near,google_robot_place_apple_in_closed_top_drawer 200-299 google_robot
add g3w3 expand9_to300 "$ART_EXPAND" widowx_carrot_on_plate 200-299 widowx_bridge
add g3w4 expand9_to300 "$ART_EXPAND" widowx_put_eggplant_in_basket,widowx_spoon_on_towel,widowx_stack_cube 200-299 widowx_bridge

sessions=()
for gpu in 1 2 3; do
  for slot in 1 2 3 4; do sessions+=("svla4700_g${gpu}_w${slot}"); done
done
for session in "${sessions[@]}"; do
  tmux has-session -t "$session" 2>/dev/null && { echo "tmux session already exists: $session" >&2; exit 2; }
done
for artifact in "$ART_DRAWERS" "$ART_RTFULL" "$ART_EXPAND"; do
  [[ ! -e "$artifact/CONFIG_LOCK.json" ]] || { echo "refusing existing protocol lock: $artifact/CONFIG_LOCK.json" >&2; exit 2; }
done
calculated=$((2*100*2 + 7*100 + 9*200*2))
[[ "$calculated" -eq 4700 && ${#sessions[@]} -eq 12 ]] || { echo "manifest count error: episodes=$calculated sessions=${#sessions[@]}" >&2; exit 2; }

for gpu in 1 2 3; do
  for slot in 1 2 3 4; do
    key="g${gpu}w${slot}"
    session="svla4700_${key}"
    worker="$ART_EXPAND/logs/${session}.sh"
    log="$ART_EXPAND/logs/${session}.launcher.log"
    mkdir -p "$(dirname "$worker")"
    {
      echo '#!/usr/bin/env bash'
      echo 'set -euo pipefail'
      printf 'export PYTHONPATH=%q HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json\n' "$PYTHONPATH"
      printf 'ROOT=%q TOKEN_CD=%q PYTHON=%q RUNNER=%q SNAPSHOTS=%q GPU=%q WORKER=%q\n' "$ROOT" "$TOKEN_CD" "$PYTHON" "$RUNNER" "$SNAPSHOTS" "$gpu" "$key"
      cat <<'SH'
run_job() {
  local study=$1 artifact=$2 tasks=$3 seeds=$4 setup=$5
  local log="$artifact/logs/${study}_${tasks//,/_}_${seeds}_gpu${GPU}_${WORKER}.log"
  mkdir -p "$artifact/logs"
  echo "[$(date -Is)] START study=$study tasks=$tasks seeds=$seeds setup=$setup gpu=$GPU"
  "$PYTHON" -u "$RUNNER" --study "$study" --artifact "$artifact" \
    --snapshot-artifact "$SNAPSHOTS" --gpu "$GPU" --worker-id "$WORKER" \
    --tasks "$tasks" --seeds "$seeds" --model-setup "$setup" >"$log" 2>&1
  echo "[$(date -Is)] DONE study=$study tasks=$tasks seeds=$seeds log=$log"
}
SH
      printf '%b' "${BODY[$key]}"
      echo "echo 'worker complete: $session'"
    } > "$worker"
    chmod +x "$worker"
  done
done

for gpu in 1 2 3; do
  for slot in 1 2 3 4; do
    key="g${gpu}w${slot}"
    session="svla4700_${key}"
    worker="$ART_EXPAND/logs/${session}.sh"
    log="$ART_EXPAND/logs/${session}.launcher.log"
    tmux new-session -d -s "$session" "bash '$worker' > '$log' 2>&1"
    echo "started $session GPU=$gpu worker_script=$worker"
  done
done

echo "started 12 workers (4 each on GPUs 1-3); planned new episodes: 400 + 700 + 3600 = 4700"
