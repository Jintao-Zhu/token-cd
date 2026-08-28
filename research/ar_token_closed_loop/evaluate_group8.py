from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, torch
from research.ar_token_counterfactual.intervention import action_logit_metrics, clean_action_token_ids, teacher_forced_forward, tensor_sha256
from research.ar_token_counterfactual.libero_runtime import build_prompt, load_policy, prepare_agentview, set_determinism
from .common import make_env, read_jsonl, write_json

def main():
 p=argparse.ArgumentParser(); p.add_argument('--workspace',type=Path,required=True); p.add_argument('--artifact',type=Path,required=True); p.add_argument('--shard-index',type=int,default=0); p.add_argument('--shard-count',type=int,default=1); a=p.parse_args(); w,out=a.workspace.resolve(),a.artifact.resolve(); src=w/'artifacts/ar_token_closed_loop_magnitude_calibration_v1_20260809_011155'; plans=[x for i,x in enumerate(read_jsonl(out/'snapshot_plan.jsonl')) if i%a.shard_count==a.shard_index]
 mean=torch.load(w/'artifacts/ar_token_counterfactual_qualification_v1_20260808_231044/position_conditioned_visual_mean.pt',map_location='cpu',weights_only=True)['mean']; ck=w/'checkpoints/openvla-7b-finetuned-libero-spatial/962318cec55ac10993ff0f5f43eda9a270b4c873'; model,proc=load_policy(ck,w/'third_party/openvla/prismatic/extern/hf'); set_determinism(7)
 from libero.libero import benchmark,get_libero_path
 from libero.libero.envs import OffScreenRenderEnv
 suite=benchmark.get_benchmark_dict()['libero_spatial']()
 for p in plans:
  path=out/'groups'/f"{p['snapshot_id']}.json"
  if path.exists(): continue
  rec=json.loads((out/'snapshots'/p['snapshot_id']/'record.json').read_text()); state=np.load(out/rec['sim_state_path'],allow_pickle=False); task=suite.get_task(p['task_id']); env=make_env(task,get_libero_path,OffScreenRenderEnv)
  try: env.reset(); obs=env.set_init_state(state); restored=np.asarray(env.get_sim_state()).copy(); _,image=prepare_agentview(obs)
  finally: env.close()
  if tensor_sha256(torch.tensor(restored))=='' or restored.shape!=state.shape or not np.array_equal(restored,state): raise RuntimeError('state restore failed')
  inputs=proc(build_prompt(task.language),image).to(model.device,dtype=torch.bfloat16); clean_ids=clean_action_token_ids(model,inputs); clean=teacher_forced_forward(model,inputs,clean_ids)
  singles=json.loads((src/'candidates'/f"{p['snapshot_id']}.json").read_text())['effects']; order=sorted(singles,key=lambda x:(-x['mean_teacher_forced_js'],x['token_index'])); top=[x['token_index'] for x in order[:8]]; bottom=[x['token_index'] for x in order[-8:]]; seed=20260809+int(hashlib.sha256(p['snapshot_id'].encode()).hexdigest()[:8],16); eligible=[i for i in range(256) if i not in top+bottom]; random=sorted(np.random.default_rng(seed).choice(eligible,8,replace=False).tolist())
  selected={};
  for name,indices in [('mask_top8_effect',top),('mask_bottom8_effect',bottom),('mask_random8',random)]:
   masked=teacher_forced_forward(model,inputs,clean_ids,indices,mean); m=action_logit_metrics(clean.logits,masked.logits); js=np.asarray(m['js_div'])[0];
   if masked.trace.changed_indices!=tuple(sorted(indices)) or not np.isfinite(js).all(): raise RuntimeError('group intervention integrity failed')
   selected[name]={'token_indices':indices,'mean_teacher_forced_js':float(js.mean()),'per_action_js':js.tolist(),'masked_projector_sha256':tensor_sha256(masked.trace.after)}
  write_json(path,{**p,'task_description':task.language,'state_sha256':rec['sim_state_sha256'],'clean_projector_sha256':tensor_sha256(clean.trace.before),'selected':selected,'all_finite':True})
  print(json.dumps({'snapshot_id':p['snapshot_id'],'complete':True}),flush=True)
if __name__=='__main__': main()
