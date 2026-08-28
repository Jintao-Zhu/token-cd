#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import h5py,numpy as np,torch,yaml
WS=Path(__file__).resolve().parents[2];sys.path[:0]=[str(WS/'LIBERO'),str(WS/'lerobot/src'),str(WS)]
from lerobot.envs.factory import make_env_pre_post_processors
from libero.libero.envs.utils import postprocess_model_xml
from research.coreact_closed_loop.runtime import env_config,load_policy_and_processors
from research.coreact_expert_direction.direction_audit import array_sha,evaluate_unit,prepare_demo_state,rewrite_demo_xml
from research.coreact_region.segmented_runtime import make_segmented_env
def atomic(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,sort_keys=True)+'\n');temp.replace(path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();w=a.workspace.resolve();art=a.artifact.resolve();q=json.loads((art/'qualification_summary.json').read_text());
 if q['decision']!='EXPERT_DIRECTION_QUALIFIED_READY_FOR_CAPTURE':raise RuntimeError('qualification gate')
 protocol=yaml.safe_load((art/'protocol.lock.yaml').read_text());units=[json.loads(x) for x in (art/'unit_manifest.jsonl').read_text().splitlines()];os.chdir(w/'LIBERO');cfg,policy,pre,_=load_policy_and_processors(w);means=torch.load(w/'artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt',weights_only=True,map_location='cpu');complete=0
 for task in range(10):
  task_units=[x for x in units if x['task_id']==task];env_pre,_=make_env_pre_post_processors(env_cfg=env_config('libero_spatial',task),policy_cfg=cfg);env=make_segmented_env('libero_spatial',task)
  try:
   with h5py.File(task_units[0]['demo_path'],'r') as f:
    for ordinal in range(50):
     group=[x for x in task_units if x['demo_ordinal']==ordinal];paths=[art/'units'/f"{x['unit_id']}.json" for x in group]
     if all(x.exists() for x in paths):complete+=3;continue
     if any(x.exists() for x in paths):raise RuntimeError(f'partial state task={task} demo={ordinal}')
     row=group[0];episode=f['data'][row['demo_id']];states=np.asarray(episode['states']);actions=np.asarray(episode['actions']);xml=episode.attrs['model_file'];xml=xml.decode() if isinstance(xml,bytes) else xml;env._env.reset();env._env.reset_from_xml_string(postprocess_model_xml(rewrite_demo_xml(xml,w),{},demo_generation=False));env._env.env.sim.reset();raw=env._env.regenerate_obs_from_state(states[row['resolved_frame']]);batch=prepare_demo_state(policy,pre,env_pre,env,raw,row['language'],actions,row['resolved_frame']);identity={'sim_state_sha256':array_sha(states[row['resolved_frame']]),'camera1_sha256':array_sha(raw['agentview_image']),'camera2_sha256':array_sha(raw['robot0_eye_in_hand_image']),'preprocessing_sha256':batch['preprocessing_sha256'],'valid_action_steps':int((~batch['action_is_pad']).sum()),'action_padding_steps':int(batch['action_is_pad'].sum())}
     records=[]
     for unit in group:
      result=evaluate_unit(policy.model,batch,means,unit['noise_seed'],unit['random_selection_seed']);records.append(({**unit,**identity,**result},art/'units'/f"{unit['unit_id']}.json"))
     if len({x[0]['prefix_sha256'] for x in records})!=1 or not all(x[0]['finite'] for x in records):raise RuntimeError(f'unit integrity task={task} demo={ordinal}')
     for value,path in records:atomic(path,value)
     complete+=3;print(json.dumps({'units_complete':complete,'planned':1500,'task':task,'demo':ordinal}),flush=True)
  finally:env.close()
 if complete==1500:(art/'status'/'capture.complete').write_text('1500/1500 matched units complete\n')
if __name__=='__main__':main()
