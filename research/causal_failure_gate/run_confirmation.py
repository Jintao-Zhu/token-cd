from __future__ import annotations
import argparse,json,os,sys,traceback
from pathlib import Path
import numpy as np,torch
_WS=Path(__file__).resolve().parents[2];sys.path[:0]=[str(_WS/'LIBERO'),str(_WS/'lerobot/src'),str(_WS)]
from lerobot.envs.factory import make_env_pre_post_processors
from research.coreact_closed_loop.runtime import env_config,prepare
from research.coreact_trained_weak.runtime import load_policy
from research.causal_failure_gate.run_screening import atomic,native_seed,run_arm,sample

def confirm_seed(meta,k): return 970_000_000+meta['task_id']*1_000_000+meta['init_state_id']*10_000+int(round(meta['target_progress']*100))*100+k*1000
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--task-id',type=int,required=True);a=p.parse_args();ws=a.workspace.resolve();art=a.artifact.resolve();os.environ['MUJOCO_GL']='egl';ck=ws/'artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522/training_run/trajectory/checkpoints/015000/pretrained_model';config,policy,pre,post=load_policy(ck)
 for screen_path in sorted((art/'screen').glob(f'task{a.task_id:02d}__*.json')):
  screen=json.loads(screen_path.read_text());sel=screen.get('selected_rescue');out=art/'confirmation'/screen_path.name
  if sel is None or out.exists(): continue
  try:
   snap=torch.load(art/'snapshots'/f'{screen_path.stem}.pt',weights_only=False,map_location='cpu');m=snap['metadata']
   if sel['source']=='expert': rescue=np.load(art/'neighbors'/f'{screen_path.stem}.npz')['actions'][sel['candidate_index']]
   else:
    ep,_=make_env_pre_post_processors(env_cfg=env_config('libero_spatial',m['task_id']),policy_cfg=config);batch=prepare(policy,pre,ep,snap['observation'],m['instruction']);rescue=sample(policy,config,batch,native_seed(m,sel['candidate_index']))
   pairs=[]
   for k in range(5):
    seed=confirm_seed(m,k);bad=run_arm(snap,np.asarray(snap['A_bad']),config,policy,pre,post,seed);good=run_arm(snap,np.asarray(rescue,np.float32),config,policy,pre,post,seed);pairs.append({'seed_index':k,'downstream_seed_base':seed,'bad_success':bad['success'],'rescue_success':good['success'],'bad_steps':bad['steps'],'rescue_steps':good['steps']})
   pbad=sum(x['bad_success'] for x in pairs)/5;pres=sum(x['rescue_success'] for x in pairs)/5;du=pres-pbad;native_dist=[x['distance'] for x in screen['candidates'] if x['source']=='native'];dn=float(np.median(native_dist));ratio=float(sel['distance']/dn) if dn>0 else float('inf');causal=pbad<=.4 and pres>=.8 and du>=.4;hard=causal and ratio<=1.0
   atomic(out,{'snapshot_id':screen_path.stem,'task_id':m['task_id'],'init_state_id':m['init_state_id'],'progress':m['target_progress'],'selected_rescue':sel,'pairs':pairs,'P_bad':pbad,'P_rescue':pres,'delta_u':du,'D_native':dn,'r':ratio,'causal_failure':causal,'causal_hard_failure':hard})
   print(json.dumps({'snapshot':screen_path.stem,'P_bad':pbad,'P_rescue':pres,'delta_u':du,'r':ratio,'causal':causal,'hard':hard}),flush=True)
  except Exception as e:
   atomic(art/'invalid_units'/f'{screen_path.stem}__confirmation.json',{'snapshot_id':screen_path.stem,'error':str(e),'traceback':traceback.format_exc()});print(json.dumps({'snapshot':screen_path.stem,'error':str(e)}),flush=True)
if __name__=='__main__':main()
