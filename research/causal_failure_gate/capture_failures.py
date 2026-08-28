from __future__ import annotations

import argparse, copy, hashlib, json, os, sys
from pathlib import Path
import numpy as np
import torch
_WS=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(_WS/'LIBERO'),str(_WS/'lerobot/src'),str(_WS)]
from lerobot.envs.factory import make_env_pre_post_processors
from research.coreact_closed_loop.runtime import env_config, prepare
from research.coreact_closed_loop.run_pilot import vector_info_value
from research.coreact_trained_weak.runtime import load_policy

HORIZON = 520
TARGETS = (0.25, 0.50, 0.75, 0.90)

def sha_array(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--workspace',type=Path,required=True); p.add_argument('--artifact',type=Path,required=True); p.add_argument('--task-ids',type=int,nargs='+',default=list(range(10))); p.add_argument('--init-state-ids',type=int,nargs='+',default=list(range(50))); a=p.parse_args()
    os.environ.setdefault('MUJOCO_GL','egl'); ws=a.workspace.resolve(); art=a.artifact.resolve()
    os.environ['HF_HOME']=str(ws/'task1/.hf-cache'); os.environ['TRANSFORMERS_CACHE']=str(ws/'task1/.hf-cache/hub'); os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    ck=ws/'artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522/training_run/trajectory/checkpoints/015000/pretrained_model'
    config, policy, pre, post = load_policy(ck)
    (art/'trajectories').mkdir(exist_ok=True)
    for task in a.task_ids:
      cfg=env_config('libero_spatial',task); ep,eop=make_env_pre_post_processors(env_cfg=cfg,policy_cfg=config)
      for init in a.init_state_ids:
        key=f'task{task:02d}__init{init:03d}'; out=art/'trajectories'/f'{key}.pt'
        if out.exists(): continue
        env=cfg.create_envs(n_envs=1,use_async_envs=False)['libero_spatial'][task]; inner=env.envs[0]
        seed=930000000+task*10000+init; actions=[]; boundaries=[]; queue=[]; success=False; language=''
        try:
          inner.init_state_id=init; obs,_=env.reset(seed=seed); language=inner.task_description; replan=0
          for step in range(HORIZON):
            if not queue:
              state=np.asarray(inner._env.get_sim_state()).copy(); xml=inner._env.sim.model.get_xml()
              batch=prepare(policy,pre,ep,obs,language); gen=torch.Generator(device=batch['state'].device).manual_seed(931000000+task*100000+init*100+replan)
              noise=torch.randn((1,config.chunk_size,config.max_action_dim),generator=gen,device=batch['state'].device,dtype=batch['state'].dtype)
              with torch.inference_mode(): chunk=policy.model.sample_actions(batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],batch['state'],noise=noise)
              if not bool(torch.isfinite(chunk).all()): raise RuntimeError('nonfinite chunk')
              boundaries.append({'control_step':step,'state':state,'xml':xml,'observation':copy.deepcopy(obs),'action_chunk':chunk[0,:,:7].detach().float().cpu().numpy(),'noise_seed':931000000+task*100000+init*100+replan,'sim_sha256':sha_array(state)})
              queue=[x.detach().cpu() for x in chunk[:,:10,:7].transpose(0,1)]; replan+=1
            ma=queue.pop(0); actions.append(ma[0].detach().float().cpu().numpy()); physical=post(ma); legal=eop({'action':physical})['action'].detach().cpu().numpy(); obs,_,term,_,info=env.step(legal); success=bool(vector_info_value(info,'is_success'))
            if success or bool(term[0]): break
          record={'task_id':task,'init_state_id':init,'reset_seed':seed,'instruction':language,'success':success,'control_steps':len(actions),'replans':replan,'actions':np.asarray(actions,np.float32),'boundaries':boundaries}
          torch.save(record,out)
          print(json.dumps({'task':task,'init':init,'success':success,'steps':len(actions),'boundaries':len(boundaries)}),flush=True)
        finally: env.close()

if __name__=='__main__': main()
