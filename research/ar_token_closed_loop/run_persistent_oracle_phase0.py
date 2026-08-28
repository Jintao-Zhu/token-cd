from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
WS=Path(__file__).resolve().parents[2];sys.path[:0]=[str(WS/'LIBERO'),str(WS)]
from research.ar_token_counterfactual.intervention import clean_action_token_ids,decode_action_ids,masked_action_token_ids,teacher_forced_forward,tensor_sha256
from research.ar_token_counterfactual.libero_runtime import build_prompt,load_policy,predict_action,prepare_agentview,prepare_env_action,set_determinism
from research.ar_token_closed_loop.common import array_sha256,make_env,write_json
ARMS=('V','A4','A8','A16','R4','R8','R16')
def seed_for(task,init,step,arm):return int(hashlib.sha256(f'{task}:{init}:{step}:{arm}'.encode()).hexdigest()[:16],16)%(2**32)
def choose(scores,arm,task,init,step):
 k=int(arm[1:]);
 if arm.startswith('A'):return np.argsort(-scores,kind='stable')[:k].tolist()
 return sorted(np.random.default_rng(seed_for(task,init,step,arm)).choice(256,k,replace=False).tolist())
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1);a=p.parse_args();w=a.workspace.resolve();art=a.artifact.resolve();mean=torch.load(w/'artifacts/ar_token_counterfactual_qualification_v1_20260808_231044/position_conditioned_visual_mean.pt',map_location='cpu',weights_only=True)['mean'];model,proc=load_policy(w/'checkpoints/openvla-7b-finetuned-libero-spatial/962318cec55ac10993ff0f5f43eda9a270b4c873',w/'third_party/openvla/prismatic/extern/hf');set_determinism(7)
 from libero.libero import benchmark,get_libero_path
 from libero.libero.envs import OffScreenRenderEnv
 suite=benchmark.get_benchmark_dict()['libero_spatial'](); plans=[(t,i) for t in range(10) for i in (0,1)];plans=[x for n,x in enumerate(plans) if n%a.shard_count==a.shard_index]
 for task_id,init in plans:
  pair=f'task{task_id:02d}__init{init:02d}';dest=art/'phase0'/f'{pair}.json'
  if dest.exists():continue
  task=suite.get_task(task_id);arms={};reference=None
  for arm in ARMS:
   env=make_env(task,get_libero_path,OffScreenRenderEnv)
   try:
    env.reset();obs=env.set_init_state(suite.get_task_init_states(task_id)[init]);
    for _ in range(10):obs,_,done,_=env.step([0,0,0,0,0,0,-1])
    state_hash=array_sha256(np.asarray(env.get_sim_state()).copy());raw,pil=prepare_agentview(obs);inputs=proc(build_prompt(task.language),pil).to(model.device,dtype=torch.bfloat16);input_hash=tensor_sha256(inputs['pixel_values']);clean=clean_action_token_ids(model,inputs);tf=teacher_forced_forward(model,inputs,clean,record_attention=True);scores=tf.attention_scores.numpy()
    if arm=='V':ids=clean;selected=[];trace=None;raw_action=predict_action(model,proc,pil,task.language);parity=float(np.max(np.abs(raw_action-decode_action_ids(model,clean))))
    else:selected=choose(scores,arm,task_id,init,0);ids,trace=masked_action_token_ids(model,inputs,selected,mean);ids2,trace2=masked_action_token_ids(model,inputs,selected,mean);parity=None
    identity=(state_hash,input_hash,tensor_sha256(clean));reference=reference or identity
    arms[arm]={'state_sha256':state_hash,'input_sha256':input_hash,'clean_ids_sha256':tensor_sha256(clean),'action_token_ids':ids[0].cpu().tolist(),'action_sha256':tensor_sha256(ids),'selected_indices':selected,'selected_count':len(selected),'changed_indices':list(trace.changed_indices) if trace else [],'repeat_deterministic':True if arm=='V' else bool(torch.equal(ids,ids2) and trace.changed_indices==trace2.changed_indices),'vanilla_decode_parity_max_abs':parity,'matched_reference':identity==reference,'all_finite':bool(torch.isfinite(ids.float()).all())}
   finally:env.close()
  diversity=len({arms[x]['action_sha256'] for x in ARMS});passed=all(x['matched_reference'] and x['repeat_deterministic'] and x['all_finite'] and (x['selected_count']==int(k[1:]) and sorted(x['selected_indices'])==sorted(x['changed_indices']) if k!='V' else x['vanilla_decode_parity_max_abs']<=1e-6) for k,x in arms.items())
  write_json(dest,{'pair_id':pair,'task_id':task_id,'init_state_id':init,'pass':passed,'distinct_action_hashes':diversity,'arms':arms});print(json.dumps({'pair':pair,'pass':passed,'diversity':diversity}),flush=True)
if __name__=='__main__':main()
