#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/code
ARTIFACT="$1"
PYTHON="$ROOT/task1/.conda-envs/flow-vla/bin/python"
SOURCE="$ROOT/artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export HF_HOME="$SOURCE/hf_home"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT/research/worktrees/lerobot_trained_weak_clean_e40b58a/src"
cd "$ROOT"

run_one() {
  local name="$1"
  printf '{"stage":"TRAINING","active":"%s","steps":15000}\n' "$name" > "$ARTIFACT/status/current.json"
  "$PYTHON" -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=lerobot/libero \
    --dataset.root="$SOURCE/dataset_cache/lerobot/libero" \
    --dataset.revision=a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 \
    --dataset.use_imagenet_stats=false \
    --dataset.video_backend=pyav \
    --output_dir="$ARTIFACT/training/$name" \
    --job_name="capacity_${name}_v1" \
    --policy.path="$ARTIFACT/initializations/$name" \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --seed=1729 \
    --steps=15000 \
    --save_checkpoint=true \
    --save_freq=5000 \
    --batch_size=32 \
    --num_workers=4 \
    --env_eval_freq=0 \
    --eval_steps=0 \
    --log_freq=50 \
    --wandb.enable=false \
    --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
    > "$ARTIFACT/logs/${name}_training.log" 2>&1
  for step in 005000 010000 015000; do
    test -f "$ARTIFACT/training/$name/checkpoints/$step/pretrained_model/model.safetensors"
  done
}

run_one weak_a_8l
run_one weak_b_4l
printf '{"stage":"SELECTION_RUNNING","states":250}\n' > "$ARTIFACT/status/current.json"
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="$SOURCE/libero_config"
export PYTHONPATH="$ROOT/research/worktrees/lerobot_trained_weak_clean_e40b58a/src:$ROOT/LIBERO:$ROOT"
"$PYTHON" -m research.coreact_capacity_weak.run_offline --workspace "$ROOT" --artifact "$ARTIFACT" --split selection > "$ARTIFACT/logs/selection.log" 2>&1
"$PYTHON" -m research.coreact_capacity_weak.analyze --artifact "$ARTIFACT" --split selection > "$ARTIFACT/logs/selection_analysis.log" 2>&1
if [[ -f "$ARTIFACT/selected_weak.lock.json" ]]; then
  printf '{"stage":"CONFIRMATION_RUNNING","states":250}\n' > "$ARTIFACT/status/current.json"
  "$PYTHON" -m research.coreact_capacity_weak.run_offline --workspace "$ROOT" --artifact "$ARTIFACT" --split confirmation > "$ARTIFACT/logs/confirmation.log" 2>&1
  "$PYTHON" -m research.coreact_capacity_weak.analyze --artifact "$ARTIFACT" --split confirmation > "$ARTIFACT/logs/confirmation_analysis.log" 2>&1
fi
decision=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "$ARTIFACT/decision.json")
printf '{"stage":"COMPLETE","decision":"%s"}\n' "$decision" > "$ARTIFACT/status/current.json"
