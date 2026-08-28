#!/usr/bin/env python3
"""Build segmentation-grounded object/interaction/background group manifest."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, torch, yaml
from lerobot.envs.factory import make_env_pre_post_processors
from research.coreact_closed_loop.run_pilot import read_jsonl,write_json
from research.coreact_closed_loop.runtime import env_config,load_policy_and_processors,prepare
from research.coreact_region.audit_effect_candidates import restore
from research.coreact_region.effect_existence import array_sha256
from research.coreact_region.fixed_mask_sampler import prepare_ranked_prefix
from research.coreact_region.region_mapping import attach_prefix_indices,instance_label_map,protected_instance_names,token_regions
from research.coreact_region.segmented_runtime import batched_observation,make_segmented_env,raw_observation
from research.coreact_closed_loop.guidance import tensor_sha256

def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();w,o=a.workspace.resolve(),a.artifact.resolve()
 cfg,policy,pre,_=load_policy_and_processors(w); means=torch.load(w/'artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt',weights_only=True,map_location='cpu')['visual_position_mean']; audits=[]; manifest=[]; failures=[]
 seen=set()
 for spec in read_jsonl(o/'source_snapshot_manifest.jsonl'):
  sid=spec['snapshot_id'];
  if sid in seen: continue
  seen.add(sid); env=make_segmented_env(spec['suite'],spec['task_id']); ep,_=make_env_pre_post_processors(env_cfg=env_config(spec['suite'],spec['task_id']),policy_cfg=cfg)
  try:
   env.init_state_id=spec['init_state_id']; env.reset(seed=spec['reset_seed']); state=np.load(o/spec['state_path'],allow_pickle=False); obs=restore(env,state)
   if array_sha256(np.asarray(env._env.get_sim_state()))!=spec['sim_state_sha256']: raise RuntimeError('state restoration mismatch')
   prepared=prepare(policy,pre,ep,batched_observation(obs),spec['language']); gen=torch.Generator(device=prepared['state'].device).manual_seed(spec['proposal_seed']); noise=torch.randn((1,cfg.chunk_size,cfg.max_action_dim),generator=gen,device=prepared['state'].device,dtype=prepared['state'].dtype)
   with torch.inference_mode(): ranked=prepare_ranked_prefix(policy.model,prepared['images'],prepared['image_masks'],prepared['lang_tokens'],prepared['lang_masks'],prepared['state'],noise)
   instances=env._env.env.model.instances_to_ids; labels=instance_label_map(instances); protected=protected_instance_names(instances); target_name=env._env.env.parsed_problem['goal_state'][0][1]
   all_regions=[]
   for cam,key in (('camera1','agentview_segmentation_instance'),('camera2','robot0_eye_in_hand_segmentation_instance')):
    regions=token_regions(raw_observation(env)[key],camera_id=cam,label_by_name=labels,relevant_names={target_name},protected_names=protected,relevant_threshold=0.0); all_regions.extend(regions)
   mapped=attach_prefix_indices(ranked['span_map'],all_regions)
   eligible=[r for r in mapped if r['protected_fraction']==0 and r['region']!='protected']; target=[r['prefix_index'] for r in eligible if r['relevant_fraction']>0]; background=[r['prefix_index'] for r in eligible if r['relevant_fraction']==0]
   target=set(target); bg=list(sorted(set(background))); interaction=set(target)
   for r in eligible:
    if r['prefix_index'] in target: continue
    if any(r['camera_id']==q['camera_id'] and abs(r['row']-q['row'])<=1 and abs(r['column']-q['column'])<=1 for q in eligible if q['prefix_index'] in target): interaction.add(r['prefix_index'])
   if len(target)<1 or len(bg)<len(interaction): raise RuntimeError(f'insufficient groups target={len(target)} background={len(bg)} interaction={len(interaction)}')
   rng=np.random.default_rng(spec['proposal_seed']); groups={"task_object":sorted(target),"interaction":sorted(interaction),"background_match":sorted(rng.choice(bg,size=len(interaction),replace=False).tolist())}
   audit={'snapshot_id':sid,'valid':True,'native_prefix_sha256':tensor_sha256(ranked['prefix']),'groups':groups,'target_count':len(target),'interaction_count':len(interaction),'background_count':len(bg),'mapped_count':len(mapped)}; write_json(o/'candidate_audits'/f'{sid}.json',audit); audits.append(audit)
   for rep in range(5):
    base={k:spec[k] for k in ('snapshot_id','suite','task_id','task_name','language','init_state_id','reset_seed','state_path','sim_state_sha256','assigned_phase','phase_fallback','selected_boundary')}; seed=spec['rollout_seed'] if rep==spec['replicate'] else int(823_000_000+spec['task_id']*1_000_000+spec['init_state_id']*10_000+rep*100+10)
    manifest.append({**base,'episode_id':f'structured__{sid}__clean__r{rep}','condition':'clean','group_id':None,'token_indices':[],'duration':0,'replicate':rep,'rollout_seed':seed,'candidate_audit_path':f'candidate_audits/{sid}.json'})
    for dur in (1,3):
     for name,g in groups.items(): manifest.append({**base,'episode_id':f'structured__{sid}__{name}__d{dur}__r{rep}','condition':'masked','group_id':name,'token_indices':g,'duration':dur,'replicate':rep,'rollout_seed':seed,'candidate_audit_path':f'candidate_audits/{sid}.json'})
  except Exception as exc: failures.append({'snapshot_id':sid,'error':str(exc)})
  finally: env.close()
 gate={'gate':'structured_candidate_integrity','pass':not failures and len(audits)==44 and len(manifest)==1540,'snapshots':len(audits),'rollouts':len(manifest),'failures':failures,'attention_not_used_for_selection':True}
 write_json(o/'integrity_gate.json',gate)
 if gate['pass']:
  with (o/'rollout_manifest.jsonl').open('x') as f:
   f.write(''.join(json.dumps(r,sort_keys=True)+'\n' for r in manifest))
  (o/'protocol.lock.yaml').write_text((o/'protocol.lock.yaml').read_text()+yaml.safe_dump({'integrity_gate':gate},sort_keys=False))
 print(json.dumps(gate,indent=2));
 if not gate['pass']: raise SystemExit(1)
if __name__=='__main__': main()
