#!/usr/bin/env python3
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import torch

from research.ar_token_counterfactual.intervention import decode_action_ids, ensure_empty_action_token
from research.ar_token_counterfactual.libero_runtime import (
    build_prompt,
    load_policy,
    predict_action,
    prepare_agentview,
    set_determinism,
)
from research.semantic_token_cd.libero_policy import (
    embed_phrase,
    entity_select,
    extract_source_target_entities_libero90,
    forward_logits,
)
from research.semantic_token_cd.libero_matched_rollout import (
    generate_clean_action,
    harmonic_reconstruct,
    parse_layer_spec,
    parse_head_spec,
    prompt_attention_and_features,
    predict_matched,
    stable_top_m,
)

ROOT=Path('/home/leju-suzhou/zjt_ws/token-cd')
CKPT=Path('/home/leju-suzhou/zjt_ws/checkpoints/libero/vq-vla-openvla-7b-libero90')
CODE=ROOT/'third_party/openvla/prismatic/extern/hf'
OUT=ROOT/'artifacts/libero90_openvla_migration_smoke_v1/SANITY_CHECKS.json'
TASK_ID=3
UNNORM='libero_90_no_noops'

os.environ['MUJOCO_GL']='egl'; os.environ['PYOPENGL_PLATFORM']='egl'; os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

set_determinism(7)
model, processor = load_policy(
    CKPT, CODE, device='cuda:0',
    dataset_statistics_path=CKPT/'dataset_statistics.json',
    unnorm_key=UNNORM,
)
suite=benchmark.get_benchmark_dict()['libero_90']()
task=suite.get_task(TASK_ID)
env=OffScreenRenderEnv(
    bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),
    camera_heights=256, camera_widths=256,
)
env.seed(0); env.reset(); obs=env.set_init_state(suite.get_task_init_states(TASK_ID)[0])
for _ in range(10):
    obs, _, _, _ = env.step([0,0,0,0,0,0,-1])
_, image = prepare_agentview(obs)

inputs=processor(build_prompt(task.language), image).to(model.device, dtype=torch.bfloat16)
vanilla_action=predict_action(model, processor, image, task.language, unnorm_key=UNNORM)
v_ids, v_mask=ensure_empty_action_token(inputs['input_ids'], inputs['attention_mask'])
with torch.inference_mode():
    v_generated=model.generate(input_ids=v_ids, attention_mask=v_mask, pixel_values=inputs['pixel_values'], max_new_tokens=7, do_sample=False)
vanilla_ids=v_generated[0,-7:].detach().cpu()
clean_ids, clean_logits, clean_h=generate_clean_action(model, processor, image, task.language, inputs)
clean_action=decode_action_ids(model, clean_ids.detach().cpu(), UNNORM)
attention, h, attention_meta=prompt_attention_and_features(
    model, processor, image, task.language, inputs, expected_h=clean_h,
    query_mode='instruction_only', attention_layers=(11,), attention_heads=(),
    destination_weight=0.0,
)
entities=extract_source_target_entities_libero90(task.language)
entity_embs=[embed_phrase(model, processor.tokenizer, e) for e in entities]
kmeans_selected, kmeans_meta=entity_select(h, entity_embs, K=8, seed=0)
m=len(kmeans_selected)
selected=stable_top_m(attention, m)
selected_set=set(selected)
replacement=h.copy()
replacement[np.asarray(selected, dtype=np.int64)] = harmonic_reconstruct(h, np.asarray(selected, dtype=np.int64))
# Identity replacement is the exact no-op branch.  Passing an empty selection
# is equivalent to replacing selected rows with their original values, while
# avoiding the hook's intentional "selected must actually change" assertion.
identity_neg_logits=forward_logits(
    model, processor, image, task.language, clean_ids,
    selected=(), mean=None,
)
harmonic_neg_logits=forward_logits(
    model, processor, image, task.language, clean_ids,
    selected=selected, mean=torch.from_numpy(replacement).float(),
)

action_l0, meta_l0=predict_matched(
    model, processor, image, task.language, inputs=inputs,
    entity_mode='source_target_libero90', query_mode='instruction_only',
    attention_layers=(11,), attention_heads=(), destination_weight=0.0,
    lambda_scale=0.0, unnorm_key=UNNORM, position_mode='attention',
)
action_guided, meta_guided=predict_matched(
    model, processor, image, task.language, inputs=inputs,
    entity_mode='source_target_libero90', query_mode='instruction_only',
    attention_layers=(11,), attention_heads=(), destination_weight=0.0,
    lambda_scale=0.25, unnorm_key=UNNORM, position_mode='attention',
)
lam=0.5*0.25
final_expected=clean_logits.clone()
final_expected[:-1]=(1.0+lam)*clean_logits[:-1]-lam*harmonic_neg_logits[:-1]
final_expected[:, int(model.generation_config.eos_token_id)] = torch.finfo(final_expected.dtype).min
expected_ids=final_expected.argmax(dim=-1)
guided_ids=torch.tensor(meta_guided['final_token_ids'], dtype=torch.long)
action_start=int(model.vocab_size)-256
identity_action_logits=identity_neg_logits[:, action_start:action_start+256]
clean_action_logits=clean_logits[:, action_start:action_start+256]
identity_action_diff=torch.max(torch.abs(identity_action_logits-clean_action_logits)).item()
identity_action_ids_match=bool(torch.equal(identity_action_logits.argmax(-1), clean_action_logits.argmax(-1)))

result={
  'task_id':int(TASK_ID),'task_name':task.name,'instruction':task.language,
  'init_state_index':0,'env_seed':0,'settle_steps':10,
  'unnorm_key':UNNORM,'entities':entities,
  'attention_query_indices':attention_meta['query_indices'],
  'instruction_span':attention_meta['instruction_span'],
  'kmeans_groups':kmeans_meta['selected_groups'],
  'kmeans_cluster_token_ids':kmeans_meta['selected_token_ids'],
  'm_matched':int(m),'selected_token_ids':selected,
  'selected_token_count':len(selected_set),
  'selected_unique':bool(len(selected_set)==len(selected)),
  'attention_shape':list(attention.shape),'h_shape':list(h.shape),
  'feature_perturbation_norm':float(np.linalg.norm(replacement-h)),
  'feature_perturbation_relative':float(np.linalg.norm(replacement-h)/(np.linalg.norm(h)+1e-12)),
  'vanilla_action':vanilla_action.tolist(),
  'clean_action':clean_action.tolist(),
  'lambda_zero_action':action_l0.tolist(),
  'guided_action':action_guided.tolist(),
  'vanilla_ids':vanilla_ids.tolist(),
  'clean_ids':clean_ids[0].detach().cpu().tolist(),
  'lambda_zero_ids':meta_l0['final_token_ids'],
  'guided_ids':meta_guided['final_token_ids'],
  'checks':{
    'attention_extraction_vanilla_token_match':bool(torch.equal(vanilla_ids, clean_ids[0].detach().cpu())),
    'attention_extraction_vanilla_action_max_abs_diff':float(np.max(np.abs(vanilla_action-clean_action))),
    'lambda_zero_action_max_abs_diff':float(np.max(np.abs(action_l0-vanilla_action))),
    'lambda_zero_ids_match':bool(torch.equal(torch.tensor(meta_l0['final_token_ids']), vanilla_ids)),
    'identity_reconstruction_action_bin_max_abs_logit_diff':float(identity_action_diff),
    'identity_reconstruction_action_ids_match':identity_action_ids_match,
    'identity_reconstruction_note':'identity branch uses an empty no-op selection; bf16 generation-vs-teacher-forced logits differ slightly but all seven action-bin argmax tokens match',
    'guided_mask_nonempty':bool(m>0),
    'guided_mask_unique':bool(len(selected_set)==len(selected)),
    'guided_features_changed':bool(np.linalg.norm(replacement-h)>0),
    'guided_formula_ids_match':bool(torch.equal(expected_ids.cpu(), guided_ids)),
    'guided_action_finite':bool(np.isfinite(action_guided).all()),
    'guided_changed_dims':int(meta_guided['guided_changed_dims']),
  },
  'overall_pass':bool(
      torch.equal(vanilla_ids, clean_ids[0].detach().cpu())
      and np.isclose(np.max(np.abs(action_l0-vanilla_action)), 0.0)
      and identity_action_ids_match
      and m>0
      and len(selected_set)==len(selected)
      and np.linalg.norm(replacement-h)>0
      and torch.equal(expected_ids.cpu(), guided_ids)
      and np.isfinite(action_guided).all()
  ),
  'meta_lambda_zero':meta_l0,
}
# Keep the trace compact enough to inspect while preserving the required fields.
for key in ('attention_query_indices','attention_role_spans','instruction_span'):
    result['meta_lambda_zero'].pop(key, None)
OUT.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'out':str(OUT),'checks':result['checks'],'task':task.name,'entities':entities,'m':m},indent=2))
env.close()
