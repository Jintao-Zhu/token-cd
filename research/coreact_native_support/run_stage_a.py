#!/usr/bin/env python3
import argparse,json,math,hashlib,time
from pathlib import Path
import numpy as np,torch
from lerobot.envs.factory import make_env_pre_post_processors
from research.coreact_closed_loop.runtime import load_policy_and_processors,make_task_env,prepare
from research.coreact_closed_loop.run_pilot import vector_info_value

H=280; TASKS=range(10); INITS=range(32,36); PROGRESS=(.25,.65); CANDS=range(6); SEEDS=range(6)
def sha(x):
 h=hashlib.sha256(); a=np.ascontiguousarray(x); h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()
def tsha(x): return sha(x.detach().cpu().contiguous().numpy())
def sample(policy,x,noise):
 return policy.model.sample_actions(x['images'],x['image_masks'],x['lang_tokens'],x['lang_masks'],x['state'],noise=noise)
def make_noise(x,cfg,seed):
 g=torch.Generator(device=x['state'].device).manual_seed(seed)
 return torch.randn((1,cfg.chunk_size,cfg.max_action_dim),generator=g,device=x['state'].device,dtype=x['state'].dtype)
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--mode',choices=['capture','rollout'],required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();ws=a.workspace.resolve();art=a.artifact.resolve();
 cfg,policy,pre,post=load_policy_and_processors(ws); means=[]
 if a.mode=='capture':
  for task in TASKS:
   ec=__import__('research.coreact_closed_loop.runtime',fromlist=['env_config']).env_config('libero_spatial',task);ep,eop=make_env_pre_post_processors(env_cfg=ec,policy_cfg=cfg)
   for init in INITS:
    env=ec.create_envs(n_envs=1,use_async_envs=False)['libero_spatial'][task]; actions=[]; snapshots={}; queue=[]
    try:
     inner=env.envs[0]; seed=960000000+task*1000+init;inner.init_state_id=init;obs,_=env.reset(seed=seed); language=inner.task_description
     for step in range(H):
      if not queue:
       x=prepare(policy,pre,ep,obs,language); noise=make_noise(x,cfg,961000000+task*10000+init*100+step//10); chunk=sample(policy,x,noise);queue=[z.detach().cpu() for z in chunk[:,:10,:7].transpose(0,1)]
      ma=queue.pop(0); legal=eop({'action':post(ma)})['action'];obs,_,term,_,info=env.step(legal.detach().cpu().numpy());actions.append(ma[0].detach().float().cpu().numpy());
      if bool(term[0]) or bool(vector_info_value(info,'is_success')):break
     L=len(actions)
     for bi,pr in enumerate(PROGRESS):
      step=max(1,math.floor(L*pr)); sid=f'task{task:02d}__init{init:02d}__p{bi}'
      (art/'snapshots').mkdir(exist_ok=True); (art/'snapshots'/f'{sid}.json').write_text(json.dumps({'snapshot_id':sid,'task_id':task,'init_state_id':init,'reset_seed':seed,'language':language,'trajectory_length':L,'progress':pr,'resolved_step':step,'prefix_actions':[x.tolist() for x in actions[:step]],'prefix_sha256':sha(np.asarray(actions[:step],np.float32))},indent=2)+'\n')
    finally:env.close()
  print(json.dumps({'captured':len(list((art/'snapshots').glob('*.json')))})); return
 snaps=sorted((art/'snapshots').glob('*.json')); rows=[(i,s) for i,s in enumerate(snaps) if i%a.shards==a.shard]
 for _,sp in rows:
  meta=json.loads(sp.read_text()); task=meta['task_id']; env_cfg=__import__('research.coreact_closed_loop.runtime',fromlist=['env_config']).env_config('libero_spatial',task);ep,eop=make_env_pre_post_processors(env_cfg=env_cfg,policy_cfg=cfg)
  for c in CANDS:
   for seed_role in ('selection','evaluation'):
    for continuation in range(3):
     seed_index=continuation if seed_role=='selection' else continuation+3; out=art/'episodes'/f"{meta['snapshot_id']}__C{c}__{seed_role}{continuation}.json";out.parent.mkdir(exist_ok=True)
     if out.exists():continue
     env=env_cfg.create_envs(n_envs=1,use_async_envs=False)['libero_spatial'][task];queue=[];acts=[];success=False;replans=0
     try:
      inner=env.envs[0];inner.init_state_id=meta['init_state_id'];obs,_=env.reset(seed=meta['reset_seed']);
      for pa in meta['prefix_actions']:
       legal=eop({'action':post(torch.tensor(pa).view(1,-1))})['action'];obs,_,term,_,_=env.step(legal.detach().cpu().numpy())
      branch_sim_sha=sha(np.asarray(inner._env.get_sim_state())); branch_x=prepare(policy,pre,ep,obs,meta['language']); branch_noise=make_noise(branch_x,cfg,970000000+task*100000+meta['init_state_id']*1000+c); branch=sample(policy,branch_x,branch_noise); candidate_sha=tsha(branch[:,:10,:7]); queue=[z.detach().cpu() for z in branch[:,:10,:7].transpose(0,1)]
      while len(acts)<H-meta['resolved_step']:
       if not queue:
        x=prepare(policy,pre,ep,obs,meta['language']);noise=make_noise(x,cfg,980000000+task*100000+meta['init_state_id']*1000+seed_index*10+replans);chunk=sample(policy,x,noise);queue=[z.detach().cpu() for z in chunk[:,:10,:7].transpose(0,1)];replans+=1
       ma=queue.pop(0);legal=eop({'action':post(ma)})['action'];obs,_,term,_,info=env.step(legal.detach().cpu().numpy());acts.append(ma[0].detach().float().cpu().numpy());success=bool(vector_info_value(info,'is_success'))
       if bool(term[0]) or success:break
     finally:env.close()
     out.write_text(json.dumps({'snapshot_id':meta['snapshot_id'],'candidate':c,'seed_role':seed_role,'continuation_seed':seed_index,'success':success,'control_steps':len(acts),'all_actions_finite':bool(np.isfinite(np.asarray(acts)).all()),'branch_sim_state_sha256':branch_sim_sha,'candidate_noise_sha256':tsha(branch_noise),'candidate_chunk_first10_sha256':candidate_sha})+'\n')
    print(json.dumps({'snapshot':meta['snapshot_id'],'candidate':c}),flush=True)
 print(json.dumps({'episodes':len(list((art/'episodes').glob('*.json')))}))
if __name__=='__main__':main()
