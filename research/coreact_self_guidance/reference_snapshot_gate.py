"""Collect and audit all preregistered vanilla-derived branch points."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import yaml
_WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(_WORKSPACE/'LIBERO')); sys.path.insert(0,str(_WORKSPACE/'lerobot/src')); sys.path.insert(0,str(_WORKSPACE))
from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare
from research.coreact_closed_loop.run_pilot import vector_info_value

TASKS=range(10); INITS=range(5); PROGRESS=(0.25,0.65); HORIZON=280

def digest(x):
 h=hashlib.sha256()
 if isinstance(x,dict):
  for k in sorted(x): h.update(k.encode()); h.update(digest(x[k]).encode())
 elif isinstance(x,(list,tuple)):
  for v in x: h.update(digest(v).encode())
 elif isinstance(x,torch.Tensor):
  x=x.detach().cpu().contiguous(); h.update(str(x.dtype).encode()); h.update(str(tuple(x.shape)).encode()); h.update(x.numpy().tobytes())
 elif isinstance(x,np.ndarray):
  x=np.ascontiguousarray(x); h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(x.tobytes())
 else: h.update(repr(x).encode())
 return h.hexdigest()

def file_sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def prepared_payload(batch):
 return {'images':[x.detach().cpu() for x in batch['images']],'image_masks':[x.detach().cpu() for x in batch['image_masks']],'lang_tokens':batch['lang_tokens'].detach().cpu(),'lang_masks':batch['lang_masks'].detach().cpu(),'state':batch['state'].detach().cpu()}

def fingerprints(env,obs,batch,actions):
 inner=env.envs[0]; sim=inner._env.env.sim; raw=inner._env.env._get_observations(); objects={k:np.asarray(v) for k,v in raw.items() if (k.endswith('_pos') or k.endswith('_quat')) and not k.startswith('robot0_')}
 robot=obs.get('robot_state',{})
 return {'simulator_state':digest(np.asarray(inner._env.get_sim_state())),'qpos':digest(np.asarray(sim.data.qpos)),'qvel':digest(np.asarray(sim.data.qvel)),'object_pose':digest(objects),'robot_observation':digest(robot),'camera1':digest(obs['pixels']['image']),'camera2':digest(obs['pixels']['image2']),'full_observation':digest(obs),'preprocessing':digest(prepared_payload(batch)),'action_prefix':digest(np.asarray(actions,dtype=np.float32))}

def replay_capture(workspace,config,policy,pre,post,task_id,init_id,seed,language,actions,steps):
 env,env_pre,env_post=make_task_env('libero_spatial',task_id,config); captures={}
 try:
  inner=env.envs[0]; inner.init_state_id=init_id; obs,_=env.reset(seed=seed)
  for i,a in enumerate(actions[:max(steps)],1):
   obs,_,_,_,_=env.step(np.asarray(a,dtype=np.float32)[None,:])
   if i in steps:
    batch=prepare(policy,pre,env_pre,obs,language); captures[i]={'fingerprints':fingerprints(env,obs,batch,actions[:i]),'observation':copy.deepcopy(obs),'prepared':prepared_payload(batch),'sim_state':np.asarray(inner._env.get_sim_state()).copy()}
 finally: env.close()
 return captures

def main():
 p=argparse.ArgumentParser(); p.add_argument('--workspace',type=Path,required=True); p.add_argument('--parent-artifact',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--tasks',default='0,1,2,3,4,5,6,7,8,9'); p.add_argument('--inits',default='0,1,2,3,4'); p.add_argument('--progress',default='0.25,0.65'); a=p.parse_args(); ws=a.workspace.resolve(); parent=a.parent_artifact.resolve(); out=a.output.resolve(); tasks=tuple(int(x) for x in a.tasks.split(',') if x); inits=tuple(int(x) for x in a.inits.split(',') if x); progress_points=tuple(float(x) for x in a.progress.split(',') if x)
 if out.exists(): raise FileExistsError(out)
 os.environ.setdefault('HF_HOME',str(ws/'task1/.hf-cache')); os.environ.setdefault('TRANSFORMERS_CACHE',str(ws/'task1/.hf-cache/hub')); os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'; os.environ['MUJOCO_GL']='egl'
 sys.path.insert(0,str(ws/'LIBERO')); sys.path.insert(0,str(ws/'lerobot/src')); sys.path.insert(0,str(ws))
 for d in ('snapshots','trajectories','logs','status'): (out/d).mkdir(parents=True,exist_ok=True)
 parent_protocol=yaml.safe_load((parent/'protocol.lock.yaml').read_text()); config,policy,pre,post=load_policy_and_processors(ws)
 from libero.libero import benchmark
 suite=benchmark.get_benchmark_dict()['libero_spatial'](); audits=[]; all_valid=True
 lock={'experiment_name':'coreact_reference_snapshot_gate_temporal_window_v1','parent_protocol':str(parent),'parent_protocol_sha256':file_sha(parent/'protocol.lock.yaml'),'selection_rule':{'episode_length':'actual vanilla trajectory length L including terminal control step','resolved_control_step':'max(1, floor(L * target_progress))','target_progress':list(progress_points),'no outcome-dependent adjustment':True},'tasks':list(tasks),'init_state_ids':list(inits),'expected_snapshots':len(tasks)*len(inits)*len(progress_points),'guidance_forbidden':True}
 (out/'reference_gate.lock.yaml').write_text(yaml.safe_dump(lock,sort_keys=False))
 for task_id in tasks:
  language=suite.get_task(task_id).language
  for init_id in inits:
   seed=120000000+task_id*100000+init_id*1000+1; env,env_pre,env_post=make_task_env('libero_spatial',task_id,config); actions=[]; queue=[]; success=False
   try:
    inner=env.envs[0]; inner.init_state_id=init_id; obs,_=env.reset(seed=seed)
    for step in range(HORIZON):
     if not queue:
      batch=prepare(policy,pre,env_pre,obs,language); gen=torch.Generator(device=batch['state'].device).manual_seed(seed*1000+step//10); noise=torch.randn((1,config.chunk_size,config.max_action_dim),generator=gen,device=batch['state'].device,dtype=batch['state'].dtype)
      with torch.inference_mode(): chunk=policy.model.sample_actions(batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],batch['state'],noise=noise)
      queue=[x.detach().cpu() for x in chunk[:,:10,:7].transpose(0,1)]
     ma=queue.pop(0); legal=env_post({'action':post(ma)})['action'].detach().cpu().numpy()[0]; obs,_,terminated,_,info=env.step(legal[None,:]); actions.append(legal.astype(np.float32)); success=bool(vector_info_value(info,'is_success'))
     if bool(terminated[0]) or success: break
   finally: env.close()
   L=len(actions); steps=[max(1,math.floor(L*x)) for x in progress_points]
   reference=replay_capture(ws,config,policy,pre,post,task_id,init_id,seed,language,actions,steps); audit=replay_capture(ws,config,policy,pre,post,task_id,init_id,seed,language,actions,steps)
   traj={'task_id':task_id,'init_state_id':init_id,'reset_seed':seed,'instruction':language,'instruction_sha256':digest(language),'trajectory_length':L,'success':success,'actions':[x.tolist() for x in actions],'action_sha256':digest(np.asarray(actions,dtype=np.float32)),'resolved_steps':steps}; (out/'trajectories'/f'task{task_id:02d}__init{init_id:02d}.json').write_text(json.dumps(traj,indent=2,sort_keys=True)+'\n')
   for bi,(progress,step) in enumerate(zip(progress_points,steps)):
    sid=f'task{task_id:02d}__init{init_id:02d}__bin{bi:02d}'; rf=reference[step]['fingerprints']; af=audit[step]['fingerprints']; fields={k:rf[k]==af[k] for k in rf}; valid=all(fields.values()); all_valid &= valid
    payload={'snapshot_id':sid,'task_id':task_id,'init_state_id':init_id,'reset_seed':seed,'trajectory_length':L,'target_progress':progress,'resolved_control_step':step,'instruction':language,'instruction_sha256':digest(language),'prefix_action_sha256':rf['action_prefix'],'reference_fingerprints':rf,'audit_fingerprints':af,'field_equal':fields,'valid':valid,'first_mismatch':next((k for k,v in fields.items() if not v),None),'checkpoint_revision':parent_protocol['checkpoint']['revision'],'checkpoint_config_sha256':parent_protocol['checkpoint']['config_sha256']}
    torch.save({'metadata':payload,'action_prefix':torch.from_numpy(np.asarray(actions[:step],dtype=np.float32)),'reference_observation':reference[step]['observation'],'reference_prepared':reference[step]['prepared'],'reference_sim_state':torch.from_numpy(reference[step]['sim_state'])},out/'snapshots'/f'{sid}.pt')
    audits.append(payload)
   print(json.dumps({'task':task_id,'init':init_id,'L':L,'steps':steps,'valid':all(x['valid'] for x in audits[-2:])}),flush=True)
 (out/'reference_audit.json').write_text(json.dumps(audits,indent=2,sort_keys=True)+'\n'); expected=len(tasks)*len(inits)*len(progress_points); decision='REFERENCE_SNAPSHOT_GATE_PASS_READY_FOR_MATCHED_CAUSAL_ROLLOUT' if all_valid and len(audits)==expected else 'REFERENCE_SNAPSHOT_GATE_FAILED_NO_CAUSAL_ROLLOUT'; (out/'decision.json').write_text(json.dumps({'decision':decision,'valid':sum(x['valid'] for x in audits),'expected':expected,'failures':[{'snapshot_id':x['snapshot_id'],'first_mismatch':x['first_mismatch']} for x in audits if not x['valid']]},indent=2)+'\n'); (out/'status'/('integrity.pass' if decision.startswith('REFERENCE_SNAPSHOT_GATE_PASS') else 'integrity.fail')).write_text(decision+'\n'); print(decision)

if __name__=='__main__': main()
