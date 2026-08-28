from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
import yaml
WS=Path(__file__).resolve().parents[2];sys.path[:0]=[str(WS/'LIBERO'),str(WS)]
from research.ar_token_counterfactual.intervention import clean_action_token_ids,decode_action_ids,masked_action_token_ids,teacher_forced_forward,tensor_sha256
from research.ar_token_counterfactual.libero_runtime import build_prompt,load_policy,predict_action,prepare_agentview,prepare_env_action,set_determinism
from research.ar_token_closed_loop.common import array_sha256,make_env,progress_dict,write_json
from research.ar_token_closed_loop.run_persistent_oracle_phase0 import choose
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1);a=p.parse_args();w=a.workspace.resolve();art=a.artifact.resolve()
 protocol=yaml.safe_load((art/'protocol.lock.yaml').read_text());arms=tuple(protocol['arms']);phase_artifact=Path(protocol.get('phase0_source_artifact',art));phase=[json.loads(x.read_text()) for x in (phase_artifact/'phase0').glob('*.json')]
 if len(phase)!=20 or not all(x['pass'] for x in phase):raise RuntimeError('phase0 gate not passed')
 mean=torch.load(w/'artifacts/ar_token_counterfactual_qualification_v1_20260808_231044/position_conditioned_visual_mean.pt',map_location='cpu',weights_only=True)['mean'];model,proc=load_policy(w/'checkpoints/openvla-7b-finetuned-libero-spatial/962318cec55ac10993ff0f5f43eda9a270b4c873',w/'third_party/openvla/prismatic/extern/hf');set_determinism(7)
 from libero.libero import benchmark,get_libero_path
 from libero.libero.envs import OffScreenRenderEnv
 suite=benchmark.get_benchmark_dict()['libero_spatial']();pairs=[(t,i) for t in protocol['tasks'] for i in protocol['init_state_ids']];pairs=[x for n,x in enumerate(pairs) if n%a.shard_count==a.shard_index]
 for task_id,init in pairs:
  pair=f'task{task_id:02d}__init{init:02d}';outputs=[art/'episodes'/f'{pair}__{arm}.json' for arm in arms]
  if all(x.exists() for x in outputs):continue
  if any(x.exists() for x in outputs):raise RuntimeError(f'partial pair {pair}')
  task=suite.get_task(task_id);records={};identities={}
  for arm in arms:
   env=make_env(task,get_libero_path,OffScreenRenderEnv);actions=[];token_chunks=[];selected_history=[];latencies=[];done=False
   try:
    env.reset();obs=env.set_init_state(suite.get_task_init_states(task_id)[init])
    for _ in range(10):obs,_,done,_=env.step([0,0,0,0,0,0,-1])
    initial_state=array_sha256(np.asarray(env.get_sim_state()).copy());initial_input=None;steps=0
    while steps<220 and not done:
     _,image=prepare_agentview(obs);started=time.perf_counter()
     inputs=proc(build_prompt(task.language),image).to(model.device,dtype=torch.bfloat16);input_hash=tensor_sha256(inputs['pixel_values'])
     if arm=='V':
      ids=clean_action_token_ids(model,inputs);raw=decode_action_ids(model,ids);selected=[]
     elif arm.startswith('R'):
      # Random selectors do not depend on a clean action or attention map.
      selected=choose(None,arm,task_id,init,steps);ids,trace=masked_action_token_ids(model,inputs,selected,mean)
      if tuple(sorted(selected))!=trace.changed_indices:raise RuntimeError('changed-index mismatch')
      raw=decode_action_ids(model,ids)
     else:
      clean=clean_action_token_ids(model,inputs);tf=teacher_forced_forward(model,inputs,clean,record_attention=True);selected=choose(tf.attention_scores.numpy(),arm,task_id,init,steps);ids,trace=masked_action_token_ids(model,inputs,selected,mean)
      if tuple(sorted(selected))!=trace.changed_indices:raise RuntimeError('changed-index mismatch')
      raw=decode_action_ids(model,ids)
     if steps==0:initial_input=input_hash
     action=prepare_env_action(raw);obs,_,done,_=env.step(action.tolist());actions.append(action.tolist());token_chunks.append(ids[0].cpu().tolist() if ids is not None else []);selected_history.append(selected);latencies.append((time.perf_counter()-started)*1000);steps+=1
   finally:env.close()
   identities[arm]=(initial_state,initial_input);arr=np.asarray(actions,dtype=np.float64);records[arm]={'pair_id':pair,'episode_id':f'{pair}__{arm}','task_id':task_id,'init_state_id':init,'arm':arm,'success':bool(done),'control_steps':steps,'initial_state_sha256':initial_state,'initial_input_sha256':initial_input,'initial_action_token_ids':token_chunks[0] if token_chunks else [],'selected_token_ids_by_step':selected_history,'trajectory_sha256':array_sha256(arr),'episode_length':steps,'mean_latency_ms':float(np.mean(latencies)) if latencies else 0.,'all_actions_finite':bool(np.isfinite(arr).all()),'persistent_intervention_calls':0 if arm=='V' else len(selected_history)}
  if len({x[0] for x in identities.values()})!=1 or len({x[1] for x in identities.values()})!=1:
   write_json(art/'invalid_pairs'/f'{pair}.json',{'pair_id':pair,'identities':identities});raise RuntimeError(f'initial parity mismatch {pair}')
  for arm in arms:write_json(art/'episodes'/f'{pair}__{arm}.json',records[arm])
  print(json.dumps({'pair':pair,'complete':True}),flush=True)
if __name__=='__main__':main()
