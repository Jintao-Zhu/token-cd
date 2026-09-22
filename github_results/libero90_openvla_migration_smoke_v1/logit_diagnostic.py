#!/usr/bin/env python3
from __future__ import annotations
import os
from pathlib import Path
import torch
from research.ar_token_counterfactual.libero_runtime import load_policy, prepare_agentview, set_determinism, build_prompt
from research.semantic_token_cd.libero_policy import forward_logits, extract_source_target_entities_libero90
from research.semantic_token_cd.libero_matched_rollout import generate_clean_action
ROOT=Path('/home/leju-suzhou/zjt_ws/token-cd'); CKPT=Path('/home/leju-suzhou/zjt_ws/checkpoints/libero/vq-vla-openvla-7b-libero90'); CODE=ROOT/'third_party/openvla/prismatic/extern/hf'
os.environ['MUJOCO_GL']='egl'; os.environ['PYOPENGL_PLATFORM']='egl'; os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
set_determinism(7)
model, processor=load_policy(CKPT,CODE,device='cuda:0',dataset_statistics_path=CKPT/'dataset_statistics.json',unnorm_key='libero_90_no_noops')
suite=benchmark.get_benchmark_dict()['libero_90'](); ti=3; task=suite.get_task(ti)
env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256)
env.seed(0); env.reset(); obs=env.set_init_state(suite.get_task_init_states(ti)[0])
for _ in range(10): obs,_,_,_=env.step([0,0,0,0,0,0,-1])
_,image=prepare_agentview(obs); inputs=processor(build_prompt(task.language),image).to(model.device,dtype=torch.bfloat16)
clean_ids, clean_logits, clean_h=generate_clean_action(model,processor,image,task.language,inputs)
identity=forward_logits(model,processor,image,task.language,clean_ids,selected=(),mean=None)
start=int(model.vocab_size)-256
a=clean_logits[:,start:start+256].float(); b=identity[:,start:start+256].float(); diff=(a-b).abs()
print('clean_ids',clean_ids.tolist())
print('identity_argmax',identity[:,start:start+256].argmax(-1).tolist())
print('clean_argmax',a.argmax(-1).tolist())
print('identity_argmax_global',identity.argmax(-1).tolist())
print('clean_argmax_global',clean_logits.argmax(-1).tolist())
print('finite_clean',torch.isfinite(clean_logits).all().item(),'finite_identity',torch.isfinite(identity).all().item())
idx=torch.nonzero(diff==diff.max())[0].tolist(); print('max_idx',idx,'max',diff.max().item())
r,c=idx; print('values_clean_identity',clean_logits[r,start+c].item(),identity[r,start+c].item())
print('per_position_max_diff',diff.max(-1).values.tolist())
print('per_position_top_clean',clean_logits[:,start:start+256].topk(3,-1).values.tolist())
print('per_position_top_identity',identity[:,start:start+256].topk(3,-1).values.tolist())
env.close()
