from __future__ import annotations
import argparse, hashlib, json, os, sys, time, traceback
from pathlib import Path
import numpy as np, torch
_WS=Path(__file__).resolve().parents[2];sys.path[:0]=[str(_WS/'LIBERO'),str(_WS/'lerobot/src'),str(_WS)]
from lerobot.envs.factory import make_env_pre_post_processors
from research.coreact_closed_loop.runtime import env_config,prepare
from research.coreact_trained_weak.runtime import load_policy
from research.coreact_region.segmented_runtime import batched_observation,make_segmented_env,step_without_autoreset
from research.coreact_region.audit_effect_candidates import restore

HORIZON=520; EXEC=10
def atomic(path,obj):
 t=path.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');t.replace(path)
def arrhash(x): return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
def native_seed(meta,k): return 950_000_000+meta['task_id']*1_000_000+meta['init_state_id']*10_000+int(round(meta['target_progress']*100))*100+k
def sample(policy,config,batch,seed):
 g=torch.Generator(device=batch['state'].device).manual_seed(int(seed));noise=torch.randn((1,config.chunk_size,config.max_action_dim),generator=g,device=batch['state'].device,dtype=batch['state'].dtype)
 with torch.inference_mode(): out=policy.model.sample_actions(batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],batch['state'],noise=noise)
 if not torch.isfinite(out).all(): raise RuntimeError('nonfinite action chunk')
 return out[0,:,:7].detach().float().cpu().numpy()
def run_arm(snap,first_chunk,config,policy,pre,post,downstream_base=None):
 m=snap['metadata'];cfg=env_config('libero_spatial',m['task_id']);ep,eop=make_env_pre_post_processors(env_cfg=cfg,policy_cfg=config);env=make_segmented_env('libero_spatial',m['task_id']);actions=[];success=False;replans=0
 try:
  env.init_state_id=m['init_state_id'];env.reset(seed=m['reset_seed']);env._env.reset_from_xml_string(snap['model_xml']);state=np.asarray(snap['sim_state']);obs=restore(env,state)
  if arrhash(np.asarray(env._env.get_sim_state()))!=m['reference_sim_sha256']: raise RuntimeError('snapshot restore mismatch')
  queue=[torch.from_numpy(np.asarray(x,np.float32))[None,:] for x in first_chunk[:EXEC]];remaining=HORIZON-m['resolved_control_step']
  for control in range(remaining):
   if not queue:
    batch=prepare(policy,pre,ep,batched_observation(obs),m['instruction']);base=int(m['original_noise_seed']) if downstream_base is None else int(downstream_base);seed=base+replans+1;chunk=sample(policy,config,batch,seed);queue=[torch.from_numpy(x)[None,:] for x in chunk[:EXEC]];replans+=1
   ma=queue.pop(0);actions.append(ma[0].numpy());legal=eop({'action':post(ma)})['action'][0].detach().cpu().numpy();obs,after=step_without_autoreset(env,legal);success=bool(after.predicate)
   if success: break
  return {'success':success,'steps':len(actions),'replans_after_intervention':replans,'actions_sha256':arrhash(np.asarray(actions,np.float32))}
 finally: env.close()
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--task-id',type=int,required=True);p.add_argument('--snapshot-id');p.add_argument('--max-new',type=int);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1);a=p.parse_args();ws=a.workspace.resolve();art=a.artifact.resolve();os.environ['MUJOCO_GL']='egl'
 ck=ws/'artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522/training_run/trajectory/checkpoints/015000/pretrained_model';config,policy,pre,post=load_policy(ck);paths=sorted((art/'snapshots').glob(f'task{a.task_id:02d}__*.pt'));paths=[x for i,x in enumerate(paths) if i%a.shard_count==a.shard_index and (a.snapshot_id is None or x.stem==a.snapshot_id)];done=0
 for path in paths:
  out=art/'screen'/f'{path.stem}.json'
  if out.exists(): continue
  if a.max_new is not None and done>=a.max_new: break
  try:
   snap=torch.load(path,weights_only=False,map_location='cpu');m=snap['metadata'];cfg=env_config('libero_spatial',m['task_id']);ep,_=make_env_pre_post_processors(env_cfg=cfg,policy_cfg=config);batch=prepare(policy,pre,ep,snap['observation'],m['instruction']);replayed=sample(policy,config,batch,m['original_noise_seed']);maxerr=float(np.max(np.abs(replayed-np.asarray(snap['A_bad']))))
   if maxerr>=1e-6: raise RuntimeError(f'A_bad replay mismatch {maxerr}')
   native=[sample(policy,config,batch,native_seed(m,k)) for k in range(8)];expert=list(np.load(art/'neighbors'/f'{path.stem}.npz')['actions']);cands=[('native',k,x,native_seed(m,k)) for k,x in enumerate(native)]+[('expert',k,x,None) for k,x in enumerate(expert)]
   bad=run_arm(snap,np.asarray(snap['A_bad']),config,policy,pre,post);rows=[]
   for source,k,chunk,seed in cands:
    result=run_arm(snap,np.asarray(chunk,np.float32),config,policy,pre,post);delta=np.asarray(chunk[:EXEC])-np.asarray(snap['A_bad'][:EXEC]);rows.append({'source':source,'candidate_index':k,'generation_seed':seed,'success':result['success'],'steps':result['steps'],'actions_sha256':result['actions_sha256'],'distance':float(np.linalg.norm(delta)),'xyz_distance':float(np.linalg.norm(delta[:,:3])),'rotation_distance':float(np.linalg.norm(delta[:,3:6])),'gripper_distance':float(np.linalg.norm(delta[:,6])),'chunk_sha256':arrhash(np.asarray(chunk,np.float32))})
   rescues=[x for x in rows if (not bad['success']) and x['success']];selected=min(rescues,key=lambda x:(x['distance'],x['source'],x['candidate_index'])) if rescues else None
   atomic(out,{'snapshot_id':path.stem,'task_id':m['task_id'],'init_state_id':m['init_state_id'],'progress':m['target_progress'],'status':'complete','A_bad_replay_max_abs_error':maxerr,'bad':bad,'candidates':rows,'screened_rescue_count':len(rescues),'selected_rescue':selected})
   done+=1;print(json.dumps({'task':m['task_id'],'snapshot':path.stem,'bad_success':bad['success'],'rescues':len(rescues),'selected':None if selected is None else [selected['source'],selected['candidate_index']]}),flush=True)
  except Exception as e:
   atomic(art/'invalid_units'/f'{path.stem}__screen.json',{'snapshot_id':path.stem,'error':str(e),'traceback':traceback.format_exc()});print(json.dumps({'snapshot':path.stem,'error':str(e)}),flush=True)
if __name__=='__main__':main()
