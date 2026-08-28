from __future__ import annotations
import argparse,json,os
from pathlib import Path
import h5py,numpy as np,torch
from scipy.spatial.transform import Rotation
from libero.libero.envs.utils import postprocess_model_xml
from research.coreact_expert_direction.direction_audit import action_chunk,rewrite_demo_xml
from research.coreact_trained_weak.run_libero10_quality_screen import env_config
from research.coreact_region.segmented_runtime import make_segmented_env
from research.coreact_why_ag_fails.build_manifold_neighbors import canonical_quaternion

GROUPS=('object_pose','eef_pose','gripper','robot_state');K=16
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--task-ids',type=int,nargs='+',required=True);p.add_argument('--k',type=int,default=K);a=p.parse_args();w,o=a.workspace.resolve(),a.artifact.resolve();source=w/'artifacts/coreact_slg_self_weak_low_noise_v1_20260816_234129/confirmation_manifold/state_manifest.jsonl';source_rows=[json.loads(x) for x in source.read_text().splitlines()];os.chdir(w/'LIBERO');manifest=[];(o/'neighbors').mkdir(exist_ok=True);(o/'neighbor_parts').mkdir(exist_ok=True)
 for task in a.task_ids:
  snapshots=sorted(o.joinpath('snapshots').glob(f'task{task:02d}__*.pt'));demo_path=Path(next(x['demo_path'] for x in source_rows if x['task_id']==task));
  if str(demo_path).startswith('/root/code/'): demo_path=w/Path(str(demo_path).removeprefix('/root/code/'))
  if not demo_path.exists(): raise FileNotFoundError(demo_path)
  env=make_segmented_env('libero_spatial',task); wrapper=env; pool=[];actions_by={}
  try:
   with h5py.File(demo_path,'r') as h:
    demos=sorted(h['data'],key=lambda x:int(x.split('_')[-1]))
    for ordinal,demo in enumerate(demos):
     ep=h['data'][demo];states=np.asarray(ep['states']);actions=np.asarray(ep['actions'],np.float32);obs=ep['obs'];actions_by[demo]=actions;xml=ep.attrs['model_file'];xml=xml.decode() if isinstance(xml,bytes) else xml;wrapper._env.reset();wrapper._env.reset_from_xml_string(postprocess_model_xml(rewrite_demo_xml(xml,w),{},demo_generation=False));inner=wrapper._env.env;body_ids=[inner.sim.model.body_name2id(obj.root_body) for obj in inner.objects]
     for frame,state in enumerate(states):
      if len(actions)-frame<50:continue
      inner.sim.set_state_from_flattened(state);inner.sim.forward();objects=np.concatenate([np.r_[inner.sim.data.body_xpos[b].copy(),canonical_quaternion(inner.sim.data.body_xquat[b])] for b in body_ids]);pool.append({'demo':demo,'ordinal':ordinal,'frame':frame,'object_pose':objects,'eef_pose':np.r_[np.asarray(obs['ee_pos'][frame]),np.asarray(obs['ee_ori'][frame])],'gripper':np.asarray(obs['gripper_states'][frame]),'robot_state':np.asarray(obs['joint_states'][frame])})
    stats={g:(np.stack([x[g] for x in pool]).mean(0),np.maximum(np.stack([x[g] for x in pool]).std(0),1e-6)) for g in GROUPS}
    for path in snapshots:
     snap=torch.load(path,weights_only=False,map_location='cpu');meta=snap['metadata'];raw_state=snap.get('reference_sim_state',snap.get('sim_state'));state=raw_state.numpy() if torch.is_tensor(raw_state) else np.asarray(raw_state);wrapper._env.reset_from_xml_string(snap['model_xml']);wrapper._env.set_state(state);wrapper._env.sim.forward();inner=wrapper._env.env;body_ids=[inner.sim.model.body_name2id(obj.root_body) for obj in inner.objects];objects=np.concatenate([np.r_[inner.sim.data.body_xpos[b].copy(),canonical_quaternion(inner.sim.data.body_xquat[b])] for b in body_ids]);ro=snap.get('reference_observation',snap.get('observation'))['robot_state'];mat=np.asarray(ro['eef']['mat'])[0];query={'object_pose':objects,'eef_pose':np.r_[np.asarray(ro['eef']['pos'])[0],Rotation.from_matrix(mat).as_rotvec()],'gripper':np.asarray(ro['gripper']['qpos'])[0],'robot_state':np.asarray(ro['joints']['pos'])[0]}
     ranked=[]
     for row in pool:
      dist=float(sum(np.mean(((row[g]-query[g])/stats[g][1])**2) for g in GROUPS));ranked.append((dist,row))
     nearest_by_demo={}
     for dist,row in sorted(ranked,key=lambda x:(x[0],x[1]['ordinal'],x[1]['frame'])):nearest_by_demo.setdefault(row['demo'],(dist,row))
     selected=sorted(nearest_by_demo.values(),key=lambda x:(x[0],x[1]['ordinal'],x[1]['frame']))[:a.k];chunks=np.stack([action_chunk(actions_by[row['demo']],row['frame'])[0].numpy() for _,row in selected]);np.savez_compressed(o/'neighbors'/f"{meta['snapshot_id']}.npz",actions=chunks);manifest.append({'snapshot_id':meta['snapshot_id'],'task_id':task,'neighbors':[{'demo':row['demo'],'frame':row['frame'],'distance':dist} for dist,row in selected]})
     print(json.dumps({'task':task,'snapshot':meta['snapshot_id']}),flush=True)
  finally:
   env.close()
 (o/'neighbor_parts'/f"tasks_{'_'.join(map(str,a.task_ids))}.json").write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
