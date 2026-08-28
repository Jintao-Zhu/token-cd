from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np, torch
_WS=Path(__file__).resolve().parents[2]; sys.path[:0]=[str(_WS/'LIBERO'),str(_WS/'lerobot/src'),str(_WS)]
from lerobot.envs.factory import make_env_pre_post_processors
from research.coreact_closed_loop.runtime import env_config
from research.coreact_trained_weak.runtime import load_policy

TARGETS=(.25,.50,.75,.90)
def digest(x): return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
def atomic(path,obj):
 t=path.with_suffix('.tmp'); t.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n'); t.replace(path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--task-id',type=int,required=True);a=p.parse_args();ws=a.workspace.resolve();art=a.artifact.resolve()
 os.environ['MUJOCO_GL']='egl';ck=ws/'artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522/training_run/trajectory/checkpoints/015000/pretrained_model';config,policy,pre,post=load_policy(ck);cfg=env_config('libero_spatial',a.task_id);ep,eop=make_env_pre_post_processors(env_cfg=cfg,policy_cfg=config)
 rows=[]
 for src in sorted((art/'trajectories').glob(f'task{a.task_id:02d}__*.pt')):
  tr=torch.load(src,weights_only=False,map_location='cpu')
  if tr['success']: continue
  selected=[]
  for progress in TARGETS:
   target=progress*tr['control_steps']; b=min(tr['boundaries'],key=lambda x:(abs(x['control_step']-target),x['control_step'])); selected.append((progress,b))
  env=cfg.create_envs(n_envs=1,use_async_envs=False)['libero_spatial'][a.task_id];inner=env.envs[0];checks={}
  try:
   inner.init_state_id=tr['init_state_id'];obs,_=env.reset(seed=tr['reset_seed']); by_step={b['control_step']:(progress,b) for progress,b in selected}
   if 0 in by_step:
    p0,b0=by_step[0];checks[p0]=digest(inner._env.get_sim_state())==b0['sim_sha256']
   for step,ma in enumerate(tr['actions'],1):
    mt=torch.from_numpy(np.asarray(ma,np.float32))[None,:];legal=eop({'action':post(mt)})['action'].detach().cpu().numpy();obs,_,term,_,_=env.step(legal)
    if step in by_step:
     pr,b=by_step[step];checks[pr]=digest(inner._env.get_sim_state())==b['sim_sha256']
    if step>=max(x[1]['control_step'] for x in selected): break
  finally: env.close()
  for idx,(progress,b) in enumerate(selected):
   sid=f"task{a.task_id:02d}__init{tr['init_state_id']:03d}__p{idx}";valid=bool(checks.get(progress,False));meta={'snapshot_id':sid,'task_id':a.task_id,'init_state_id':tr['init_state_id'],'reset_seed':tr['reset_seed'],'instruction':tr['instruction'],'trajectory_success':False,'trajectory_length':tr['control_steps'],'target_progress':progress,'resolved_control_step':b['control_step'],'reference_sim_sha256':b['sim_sha256'],'replay_sim_equal':valid,'valid':valid,'original_noise_seed':b['noise_seed']}
   payload={'metadata':meta,'sim_state':b['state'],'model_xml':b['xml'],'observation':b['observation'],'A_bad':b['action_chunk'],'action_prefix':tr['actions'][:b['control_step']]}
   torch.save(payload,art/'snapshots'/f'{sid}.pt');atomic(art/'audits'/f'{sid}.json',meta);rows.append(meta)
  print(json.dumps({'trajectory':src.stem,'snapshots':4,'valid':sum(checks.values())}),flush=True)
 atomic(art/'audits'/f'task{a.task_id:02d}_summary.json',{'task_id':a.task_id,'failure_trajectories':len(rows)//4,'snapshots':len(rows),'valid':sum(x['valid'] for x in rows)})
if __name__=='__main__':main()
