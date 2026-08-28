from __future__ import annotations
import argparse,copy,json,math,os
from pathlib import Path
import numpy as np,torch
from lerobot.envs.factory import make_env_pre_post_processors
from research.coreact_closed_loop.runtime import env_config, prepare
from research.coreact_closed_loop.run_pilot import vector_info_value
from research.coreact_self_guidance.reference_snapshot_gate import digest,fingerprints
from research.coreact_trained_weak.runtime import load_policy

DEFAULT_INITS=range(40,45);PROGRESS=(.30,.65);HORIZON=520
def atomic(path,value):
 t=path.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');t.replace(path)
def replay(config,policy,pre,post,task,init,seed,language,actions,steps):
 cfg=env_config('libero_spatial',task);ep,eop=make_env_pre_post_processors(env_cfg=cfg,policy_cfg=config);env=cfg.create_envs(n_envs=1,use_async_envs=False)['libero_spatial'][task];out={}
 try:
  inner=env.envs[0];inner.init_state_id=init;obs,_=env.reset(seed=seed)
  for i,a in enumerate(actions[:max(steps)],1):
   obs,_,term,_,_=env.step(np.asarray(a,dtype=np.float32)[None,:])
   if bool(term[0]):raise RuntimeError('prefix terminated')
   if i in steps:
    batch=prepare(policy,pre,ep,obs,language);out[i]={'fingerprints':fingerprints(env,obs,batch,actions[:i]),'observation':copy.deepcopy(obs),'sim_state':np.asarray(inner._env.get_sim_state()).copy(),'model_xml':inner._env.sim.model.get_xml()}
 finally:env.close()
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--task-ids',type=int,nargs='+',required=True);p.add_argument('--init-state-ids',type=int,nargs='+',default=list(DEFAULT_INITS));a=p.parse_args();w,o=a.workspace.resolve(),a.artifact.resolve();os.environ.setdefault('MUJOCO_GL','egl');ck=w/'artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522/training_run/trajectory/checkpoints/015000/pretrained_model';config,policy,pre,post=load_policy(ck)
 for task in a.task_ids:
  cfg=env_config('libero_spatial',task);ep,eop=make_env_pre_post_processors(env_cfg=cfg,policy_cfg=config)
  for init in a.init_state_ids:
   ids=[f'task{task:02d}__init{init:02d}__p{i}' for i in range(2)]
   if all((o/'snapshots'/f'{x}.pt').exists() for x in ids):continue
   env=cfg.create_envs(n_envs=1,use_async_envs=False)['libero_spatial'][task];queue=[];legal_actions=[];success=False;seed=930_000_000+task*1000+init
   try:
    inner=env.envs[0];inner.init_state_id=init;obs,_=env.reset(seed=seed);language=inner.task_description;replan=0
    for _ in range(HORIZON):
     if not queue:
      batch=prepare(policy,pre,ep,obs,language);g=torch.Generator(device=batch['state'].device).manual_seed(931_000_000+task*10000+init*100+replan);noise=torch.randn((1,config.chunk_size,config.max_action_dim),generator=g,device=batch['state'].device,dtype=batch['state'].dtype);chunk=policy.model.sample_actions(batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],batch['state'],noise=noise);queue=[x.detach().cpu() for x in chunk[:,:10,:7].transpose(0,1)];replan+=1
     ma=queue.pop(0);legal=eop({'action':post(ma)})['action'].detach().cpu().numpy()[0].astype(np.float32);obs,_,term,_,info=env.step(legal[None,:]);legal_actions.append(legal);success=bool(vector_info_value(info,'is_success'))
     if success or bool(term[0]):break
   finally:env.close()
   L=len(legal_actions);steps=[max(1,math.floor(L*x)) for x in PROGRESS];ref=replay(config,policy,pre,post,task,init,seed,language,legal_actions,steps);audit=replay(config,policy,pre,post,task,init,seed,language,legal_actions,steps)
   for bi,(progress,step,sid) in enumerate(zip(PROGRESS,steps,ids)):
    rf,af=ref[step]['fingerprints'],audit[step]['fingerprints'];fields={k:rf[k]==af[k] for k in rf};meta={'snapshot_id':sid,'task_id':task,'init_state_id':init,'reset_seed':seed,'instruction':language,'trajectory_length':L,'trajectory_success':success,'target_progress':progress,'resolved_control_step':step,'reference_fingerprints':rf,'audit_fingerprints':af,'field_equal':fields,'valid':all(fields.values()),'prefix_action_sha256':digest(np.asarray(legal_actions[:step],np.float32))};torch.save({'metadata':meta,'action_prefix':torch.from_numpy(np.asarray(legal_actions[:step],np.float32)),'reference_observation':ref[step]['observation'],'reference_sim_state':torch.from_numpy(ref[step]['sim_state']),'model_xml':ref[step]['model_xml']},o/'snapshots'/f'{sid}.pt');atomic(o/'audits'/f'{sid}.json',meta)
    if not meta['valid']:raise RuntimeError(f'snapshot mismatch {sid}')
   print(json.dumps({'task':task,'init':init,'trajectory_length':L,'success':success,'steps':steps}),flush=True)
if __name__=='__main__':main()
