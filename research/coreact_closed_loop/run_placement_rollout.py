#!/usr/bin/env python3
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np,torch
from lerobot.envs.factory import make_env_pre_post_processors
from research.coreact_closed_loop.guidance import GuidanceConfig,sample_coreact_actions,tensor_sha256
from research.coreact_closed_loop.run_pilot import vector_info_value
from research.coreact_closed_loop.runtime import load_policy_and_processors,env_config,prepare

ARMS=('vanilla','full','early','mid','late','near','far'); MAX_STEPS=280; EXEC=10
SPECS={'full':{},'early':{'flow_step_end':3},'mid':{'flow_step_start':3,'flow_step_end':7},'late':{'flow_step_start':7},'near':{'action_end':25},'far':{'action_start':25}}
def ah(x):
 a=np.ascontiguousarray(x);h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--arm',choices=ARMS,required=True);a=p.parse_args();ws=a.workspace.resolve();art=a.artifact.resolve();lock=json.loads((art/'protocol.lock.json').read_text());mult=lock['locked_multipliers'].get(a.arm,0)
 cfg,policy,pre,post=load_policy_and_processors(ws);means=torch.load(ws/'artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt',weights_only=True,map_location='cpu');out=art/'episodes';out.mkdir(exist_ok=True)
 for task in range(10):
  ec=env_config('libero_spatial',task);ep,eop=make_env_pre_post_processors(env_cfg=ec,policy_cfg=cfg)
  for init in range(20):
   path=out/f'task{task:02d}__init{init:02d}__{a.arm}.json'
   if path.exists():continue
   env=ec.create_envs(n_envs=1,use_async_envs=False)['libero_spatial'][task];queue=[];acts=[];traces=[];lat=[];nh=[];success=False;replans=0
   try:
    inner=env.envs[0];inner.init_state_id=init;obs,_=env.reset(seed=920000000+task*1000+init);language=inner.task_description;simhash=ah(inner._env.get_sim_state());prehash=None
    for _ in range(MAX_STEPS):
     if not queue:
      x=prepare(policy,pre,ep,obs,language)
      if prehash is None:prehash=hashlib.sha256(''.join(tensor_sha256(x[k]) for k in ('state','lang_tokens','lang_masks')).encode()).hexdigest()
      noise=torch.randn((1,50,cfg.max_action_dim),generator=torch.Generator(device=x['state'].device).manual_seed(930000000+task*100000+init*1000+replans),device=x['state'].device,dtype=x['state'].dtype);nh.append(tensor_sha256(noise));st=time.perf_counter()
      if a.arm=='vanilla':chunk=policy.model.sample_actions(x['images'],x['image_masks'],x['lang_tokens'],x['lang_masks'],x['state'],noise=noise);tr=None
      else:chunk,tr=sample_coreact_actions(policy.model,x['images'],x['image_masks'],x['lang_tokens'],x['lang_masks'],x['state'],noise,means['visual_position_mean'],config=GuidanceConfig(branch='acg',placement_multiplier=mult,**SPECS[a.arm]),selection_seed=1)
      torch.cuda.synchronize();lat.append(time.perf_counter()-st);queue.extend(chunk[:,:EXEC,:7].transpose(0,1));
      if tr:traces.append({'replan':replans,**tr})
      replans+=1
     action=queue.pop(0);assert bool(torch.isfinite(action).all());physical=post(action);legal=eop({'action':physical})['action'];obs,_,term,_,info=env.step(legal.detach().cpu().numpy());acts.append(action[0].detach().float().cpu());success=bool(vector_info_value(info,'is_success'))
     if success or bool(term[0]):break
   finally:env.close()
   aa=torch.stack(acts);tv=float(torch.linalg.vector_norm(aa[1:]-aa[:-1],dim=1).sum()) if len(aa)>1 else 0
   rec={'episode_id':path.stem,'unit_id':f'task{task:02d}__init{init:02d}','arm':a.arm,'suite':'libero_spatial','task_id':task,'init_state_id':init,'reset_seed':920000000+task*1000+init,'status':'complete','success':success,'control_steps':len(acts),'replans':replans,'initial_sim_state_sha256':simhash,'initial_prepared_input_sha256':prehash,'noise_sha256_by_replan':nh,'all_actions_finite':bool(torch.isfinite(aa).all()),'action_total_variation':tv,'median_replan_latency_seconds':float(np.median(lat)),'placement_multiplier':mult if a.arm!='vanilla' else None,'replan_traces':traces}
   tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(rec,indent=2)+'\n');tmp.replace(path);print(json.dumps({'arm':a.arm,'task':task,'init':init,'success':success,'steps':len(acts)}),flush=True)
if __name__=='__main__':main()
