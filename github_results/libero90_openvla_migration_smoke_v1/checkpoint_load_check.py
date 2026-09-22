#!/usr/bin/env python3
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import torch

from research.ar_token_counterfactual.libero_runtime import (
    build_prompt,
    load_policy,
    predict_action,
    prepare_agentview,
    set_determinism,
)
from research.semantic_token_cd.libero_policy import extract_source_target_entities_libero90

ROOT=Path('/home/leju-suzhou/zjt_ws/token-cd')
CKPT=Path('/home/leju-suzhou/zjt_ws/checkpoints/libero/vq-vla-openvla-7b-libero90')
CODE=ROOT/'third_party/openvla/prismatic/extern/hf'
OUT=ROOT/'artifacts/libero90_openvla_migration_smoke_v1/CHECKPOINT_LOAD_CHECK.json'

os.environ['MUJOCO_GL']='egl'; os.environ['PYOPENGL_PLATFORM']='egl'; os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

set_determinism(7)
model, processor = load_policy(
    CKPT, CODE, device='cuda:0',
    dataset_statistics_path=CKPT/'dataset_statistics.json',
    unnorm_key='libero_90_no_noops',
)
suite=benchmark.get_benchmark_dict()['libero_90']()
task_index=3
task=suite.get_task(task_index)
env=OffScreenRenderEnv(
    bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),
    camera_heights=256, camera_widths=256,
)
env.seed(0)
env.reset()
obs=env.set_init_state(suite.get_task_init_states(task_index)[0])
for _ in range(10):
    obs, _, _, _ = env.step([0,0,0,0,0,0,-1])
_, image = prepare_agentview(obs)
action = predict_action(model, processor, image, task.language, unnorm_key='libero_90_no_noops')
stats=model.get_action_stats('libero_90_no_noops')
result={
    'checkpoint':str(CKPT),
    'architecture':model.config.architectures,
    'n_action_bins':int(model.config.n_action_bins),
    'model_vocab_size':int(model.vocab_size),
    'norm_stats_key':'libero_90_no_noops',
    'norm_stats_keys':sorted(model.norm_stats.keys()),
    'action_q01':stats['q01'],
    'action_q99':stats['q99'],
    'action_mask':stats.get('mask'),
    'task_id':task_index,
    'task_name':task.name,
    'instruction':task.language,
    'parsed_entities':extract_source_target_entities_libero90(task.language),
    'image_shape':list(image.size),
    'action_shape':list(action.shape),
    'action':action.tolist(),
    'action_finite':bool(np.isfinite(action).all()),
}
OUT.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,indent=2,sort_keys=True))
env.close()
