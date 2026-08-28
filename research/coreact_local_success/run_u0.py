from __future__ import annotations
import argparse,hashlib,json,os
from collections import defaultdict
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F
from lerobot.envs.factory import make_env_pre_post_processors
from research.coreact_closed_loop.runtime import prepare
from research.coreact_closed_loop.run_pilot import vector_info_value
from research.coreact_self_guidance.reference_snapshot_gate import digest,fingerprints
from research.coreact_closed_loop.runtime import env_config
from research.coreact_trained_weak.runtime import load_policy
from research.coreact_why_ag_fails.run_manifold_diagnosis import normalize_candidates
from research.coreact_w1_slg_rollout.sampler import sample_w1_slg_actions

ARMS=('Strong','W1_plus','W1_minus','Nearest_manifold','Random_smooth');HORIZON=520
def atomic(p,x):t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');t.replace(p)
def smooth_random(shape,seed,device,dtype):
 g=torch.Generator(device=device).manual_seed(seed);x=torch.randn(shape,generator=g,device=device,dtype=dtype);z=x.transpose(1,2);kernel=torch.tensor([1.,2.,3.,2.,1.],device=device,dtype=dtype);kernel=(kernel/kernel.sum()).view(1,1,5).expand(z.shape[1],1,5);return F.conv1d(F.pad(z,(2,2),mode='replicate'),kernel,groups=z.shape[1]).transpose(1,2)
def scale(delta,norm):return delta*(norm/(torch.linalg.vector_norm(delta)+1e-12))
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--shard-count',type=int,default=1);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--max-units',type=int);a=p.parse_args();w,o=a.workspace.resolve(),a.artifact.resolve();os.environ.setdefault('MUJOCO_GL','egl');rows=[json.loads(x) for x in (o/'u0_manifest.jsonl').read_text().splitlines()];units=defaultdict(dict)
 for r in rows:units[r['unit_id']][r['arm']]=r
 selected=[x for i,x in enumerate(sorted(units.items())) if i%a.shard_count==a.shard_index];selected=selected[:a.max_units] if a.max_units else selected;ck=Path(json.loads((o/'protocol.lock.json').read_text())['strong_checkpoint']);config,policy,pre,post=load_policy(ck);complete=0
 for uid,specs in selected:
  outs={arm:o/'u0'/f"{specs[arm]['episode_id']}.json" for arm in ARMS}
  if all(x.exists() for x in outs.values()):continue
  if any(x.exists() for x in outs.values()):raise RuntimeError(f'partial {uid}')
  sid=specs['Strong']['snapshot_id'];snap=torch.load(o/'snapshots'/f'{sid}.pt',weights_only=False,map_location='cpu');meta=snap['metadata'];prefix=snap['action_prefix'].numpy();raw_candidates=np.load(o/'neighbors'/f'{sid}.npz')['actions'];branch={};records={}
  for arm in ARMS:
   s=specs[arm];cfg=env_config('libero_spatial',s['task_id']);ep,eop=make_env_pre_post_processors(env_cfg=cfg,policy_cfg=config);env=cfg.create_envs(n_envs=1,use_async_envs=False)['libero_spatial'][s['task_id']];queue=[];actions=[];success=False;reason='horizon';replans=0
   try:
    inner=env.envs[0];inner.init_state_id=s['init_state_id'];obs,_=env.reset(seed=meta['reset_seed'])
    for pa in prefix:
     obs,_,term,_,_=env.step(np.asarray(pa,np.float32)[None,:])
     if bool(term[0]):raise RuntimeError('prefix terminated')
    batch=prepare(policy,pre,ep,obs,meta['instruction']);fp=fingerprints(env,obs,batch,list(prefix));g=torch.Generator(device=batch['state'].device).manual_seed(s['noise_seed']);noise=torch.randn((1,config.chunk_size,config.max_action_dim),generator=g,device=batch['state'].device,dtype=batch['state'].dtype)
    with torch.inference_mode():
     strong=policy.model.sample_actions(batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],batch['state'],noise=noise);guided,_=sample_w1_slg_actions(policy.model,batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],batch['state'],noise,arm='low_w1',lambda_value=.05)
    base=strong[:,:10,:7];dw=guided[:,:10,:7]-base;target_norm=torch.linalg.vector_norm(dw);candidates=normalize_candidates(pre,raw_candidates,base.device,base.dtype,config.max_action_dim)[:,:10,:7];nearest=candidates[torch.argmin(torch.sum((candidates-base)**2,dim=(1,2)))].unsqueeze(0);dm=scale(nearest-base,target_norm);rand=scale(smooth_random(base.shape,s['random_direction_seed'],base.device,base.dtype),target_norm);delta={'Strong':torch.zeros_like(base),'W1_plus':dw,'W1_minus':-dw,'Nearest_manifold':dm,'Random_smooth':rand}[arm];intervened=base+delta;chunk=strong.clone();chunk[:,:10,:7]=intervened;branch[arm]={'fingerprint':fp,'noise':digest(noise),'strong_chunk':digest(strong),'dw_norm':float(target_norm),'delta_norm':float(torch.linalg.vector_norm(delta)),'delta_sha256':digest(delta),'nearest_index':int(torch.argmin(torch.sum((candidates-base)**2,dim=(1,2)))) if arm=='Nearest_manifold' else None};queue=[x.detach().cpu() for x in chunk[:,:10,:7].transpose(0,1)];replans=1;remaining=HORIZON-meta['resolved_control_step']
    for _ in range(remaining):
     if not queue:
      batch=prepare(policy,pre,ep,obs,meta['instruction']);g=torch.Generator(device=batch['state'].device).manual_seed(s['noise_seed']+replans);n=torch.randn((1,config.chunk_size,config.max_action_dim),generator=g,device=batch['state'].device,dtype=batch['state'].dtype);chunk=policy.model.sample_actions(batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],batch['state'],noise=n);queue=[x.detach().cpu() for x in chunk[:,:10,:7].transpose(0,1)];replans+=1
     ma=queue.pop(0);legal=eop({'action':post(ma)})['action'];obs,_,term,_,info=env.step(legal.detach().cpu().numpy());actions.append(ma[0].float().cpu());success=bool(vector_info_value(info,'is_success'))
     if success:reason='success';break
     if bool(term[0]):reason='terminated';break
   finally:env.close()
   records[arm]={**s,'status':'complete','success':success,'termination_reason':reason,'continuation_steps':len(actions),'replans':replans,'all_actions_finite':bool(actions and all(torch.isfinite(x).all() for x in actions)),'branch':branch[arm]}
  for field in ('fingerprint','noise','strong_chunk','dw_norm'):
   if len({json.dumps(branch[x][field],sort_keys=True) for x in ARMS})!=1:
    atomic(o/'invalid_units'/f'{uid}.json',{'unit_id':uid,'field':field,'branch':branch});raise RuntimeError(f'{uid} {field}')
  if abs(branch['W1_plus']['delta_norm']-branch['W1_minus']['delta_norm'])>1e-5:raise RuntimeError('W1 symmetry')
  for arm in ARMS:atomic(outs[arm],records[arm])
  complete+=1;print(json.dumps({'worker':a.shard_index,'complete':complete,'unit':uid}),flush=True)
if __name__=='__main__':main()
