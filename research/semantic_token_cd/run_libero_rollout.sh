#!/usr/bin/env bash
# Driver for LIBERO_OBJECT_SEMANTIC_ENTITY_CD_PHASE0_V1 — full rollout (10 tasks).
# Runs one process per task (4 arms x 20 episodes each), P tasks concurrently.
set -euo pipefail

cd /data/docker/dev_zjt/data/code

P="${P:-4}"          # concurrent task processes
EPISODES="$(seq -s, 0 19)"
ARTIFACT="artifacts/libero_object_semantic_entity_cd_phase0_v1"
mkdir -p "$ARTIFACT/rollout_logs"

export HF_HUB_OFFLINE=1 TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

TASKS=(
  pick_up_the_alphabet_soup_and_place_it_in_the_basket
  pick_up_the_cream_cheese_and_place_it_in_the_basket
  pick_up_the_salad_dressing_and_place_it_in_the_basket
  pick_up_the_bbq_sauce_and_place_it_in_the_basket
  pick_up_the_ketchup_and_place_it_in_the_basket
  pick_up_the_tomato_sauce_and_place_it_in_the_basket
  pick_up_the_butter_and_place_it_in_the_basket
  pick_up_the_milk_and_place_it_in_the_basket
  pick_up_the_chocolate_pudding_and_place_it_in_the_basket
  pick_up_the_orange_juice_and_place_it_in_the_basket
)

printf '%s\n' "${TASKS[@]}" | xargs -P "$P" -I {} bash -c '
  cd /data/docker/dev_zjt/data/code
  export HF_HUB_OFFLINE=1 TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="./LIBERO:$PWD"
  task="$1"; episodes="$2"
  task1/.venvs/openvla-ar/bin/python -u research/semantic_token_cd/libero_rollout.py \
    --artifact artifacts/libero_object_semantic_entity_cd_phase0_v1 \
    --task "$task" --episodes "$episodes" --gpu 0 \
    > "artifacts/libero_object_semantic_entity_cd_phase0_v1/rollout_logs/${task}.log" 2>&1
  echo "done: $task"
' _ {} "$EPISODES"
