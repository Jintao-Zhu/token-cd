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
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();w=a.workspace.resolve();art=a.artifact.resolve();protocol=yaml.safe_load((art/'protocol.lock.yaml').read_text());os.chdir(w/'LIBERO');cfg,policy,pre,_=load_policy_and_processors(w);means=torch.load(w/'artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt',weights_only=True,map_location='cpu');states=[json.loads(x) for x in (art/'state_manifest.jsonl').read_text().splitlines()];rows=[];repeat_pass=True
 for task in range(10):
  row=next(x for x in states if x['task_id']==task);env_pre,_=make_env_pre_post_processors(env_cfg=env_config('libero_spatial',task),policy_cfg=cfg);env=make_segmented_env('libero_spatial',task)
  try:
   with h5py.File(row['demo_path'],'r') as f:
    episode=f['data'][row['demo_id']];states_np=np.asarray(episode['states']);actions=np.asarray(episode['actions']);xml=episode.attrs['model_file'];xml=xml.decode() if isinstance(xml,bytes) else xml;env._env.reset();env._env.reset_from_xml_string(postprocess_model_xml(rewrite_demo_xml(xml,w),{},demo_generation=False));env._env.env.sim.reset();raw=env._env.regenerate_obs_from_state(states_np[row['resolved_frame']]);batch=prepare_demo_state(policy,pre,env_pre,env,raw,row['language'],actions,row['resolved_frame']);result=evaluate_unit(policy.model,batch,means,202608150000+task*10000,20260815+task*10000)
    exact8=all(len(result['selected_indices'][x])==8 and sorted(result['selected_indices'][x])==sorted(result['changed_indices'][x]) for x in ('attention','random'));protected=all(i in result['selected_indices'][name] for name in ('attention','random') for i in result['changed_indices'][name]);entry={'task_id':task,'state_id':row['state_id'],'sim_state_sha256':array_sha(states_np[row['resolved_frame']]),'camera1_sha256':array_sha(raw['agentview_image']),'camera2_sha256':array_sha(raw['robot0_eye_in_hand_image']),'preprocessing_sha256':batch['preprocessing_sha256'],'eligible_visual_count':result['eligible_visual_count'],'native_attention_layers':result['native_attention_layers'],'exact8_changed':exact8,'only_selected_changed':protected,'finite':result['finite'],'cached_full_tau1_max_abs':result['cached_full_tau1_max_abs'],'cached_full_parity':result['cached_full_tau1_max_abs']<=1e-4};entry['pass']=all(entry[x] for x in ('exact8_changed','only_selected_changed','finite','cached_full_parity')) and entry['eligible_visual_count']==128 and entry['native_attention_layers']==16;(art/'qualification'/f'task{task:02d}.json').write_text(json.dumps(entry,indent=2,sort_keys=True)+'\n');rows.append(entry)
    if task==0:
     result2=evaluate_unit(policy.model,batch,means,202608150000,20260815);repeat_pass=result==result2
  finally:env.close()
 gate={'tasks':len(rows),'all_tasks_pass':all(x['pass'] for x in rows),'deterministic_repeat':repeat_pass,'frozen_eval':not policy.training and not any(x.requires_grad for x in policy.parameters()),'max_cached_full_tau1_abs':max(x['cached_full_tau1_max_abs'] for x in rows)};gate['pass']=all(gate[x] for x in ('all_tasks_pass','deterministic_repeat','frozen_eval')) and gate['tasks']==10;decision='EXPERT_DIRECTION_QUALIFIED_READY_FOR_CAPTURE' if gate['pass'] else 'EXPERT_DIRECTION_QUALIFICATION_FAILED_NO_CAPTURE';(art/'qualification_summary.json').write_text(json.dumps({'decision':decision,'gate':gate,'tasks':rows},indent=2,sort_keys=True)+'\n');(art/'decision.json').write_text(json.dumps({'decision':decision,'gate':gate,'capture_started':False},indent=2)+'\n');(art/'status'/('qualification.pass' if gate['pass'] else 'qualification.fail')).write_text(decision+'\n');print(json.dumps(gate,indent=2))
if __name__=='__main__':main()
