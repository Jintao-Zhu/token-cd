#!/usr/bin/env python3
"""Replay selected matched states and compare clean vs guided action."""
from __future__ import annotations
import argparse, csv, json, os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch

from research.ar_token_counterfactual.libero_runtime import load_policy, predict_action, prepare_agentview, build_prompt
from research.semantic_token_cd.libero_matched_rollout import predict_matched, prompt_attention_and_features
from research.semantic_token_cd.libero_policy import extract_source_target_entities_libero90

GRID=16


def mask_components(tokens):
    remaining={int(x) for x in tokens}; count=0
    while remaining:
        count += 1; stack=[remaining.pop()]
        while stack:
            x=stack.pop(); r,c=divmod(x,GRID)
            for y in (x-GRID if r>0 else None, x+GRID if r<GRID-1 else None,
                      x-1 if c>0 else None, x+1 if c<GRID-1 else None):
                if y is not None and y in remaining:
                    remaining.remove(y); stack.append(y)
    return count

def atomic_json(path,payload):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)

def overlay_heatmap(image,attention):
 arr=np.asarray(image.resize((224,224)),dtype=np.float32)
 a=np.asarray(attention,dtype=np.float32).reshape(GRID,GRID); a=(a-a.min())/(a.ptp()+1e-8)
 heat=np.repeat(np.repeat(a,14,axis=0),14,axis=1)[:224,:224]
 hm=np.zeros_like(arr); hm[...,0]=255*heat; hm[...,1]=100*heat
 return Image.fromarray(np.clip(0.5*arr+0.5*hm,0,255).astype(np.uint8))

def overlay_mask(image,tokens):
 arr=np.asarray(image.resize((224,224)).copy()); out=arr.copy(); cell=14
 for token in tokens:
  r,c=divmod(int(token),GRID); out[r*cell:(r+1)*cell,c*cell:(c+1)*cell]=[255,0,0]
 out=(0.55*arr+0.45*out).astype(np.uint8)
 return Image.fromarray(out)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('case_ids',nargs='+'); a=ap.parse_args(); root=a.root.resolve(); outdir=root/'diagnostic_panels'; outdir.mkdir(parents=True,exist_ok=True)
 events={r['case_id']:r for r in csv.DictReader(open(root/'CASE_EVENT_TABLE.csv'))}
 ckpt=Path('/home/leju-suzhou/zjt_ws/checkpoints/libero/vq-vla-openvla-7b-libero90')
 model,processor=load_policy(ckpt,Path('/home/leju-suzhou/zjt_ws/token-cd/third_party/openvla/prismatic/extern/hf'),device='cuda:0',dataset_statistics_path=ckpt/'dataset_statistics.json',unnorm_key='libero_90_no_noops')
 from libero.libero import benchmark,get_libero_path
 from libero.libero.envs import OffScreenRenderEnv
 suite=benchmark.get_benchmark_dict()['libero_90'](); rows=[]
 for cid in a.case_ids:
  e=events[cid]
  # recover task id via manifest to avoid parsing names
  manifest=json.loads((root/'TASK_MANIFEST.json').read_text()); item=next(x for x in manifest['tasks'] if x['task_name']==e['task']); tid=int(item['task_id']); iid=int(e['init_state_id']); task=suite.get_task(tid); init=suite.get_task_init_states(tid)[iid]
  d=json.loads((root/'episodes'/task.name/'matched'/f'init_{iid:03d}.json').read_text()); actions=d['trajectory']; target_step=int(e['first_guided_change_step']) if e['first_guided_change_step'] else 0
  env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256)
  try:
   env.seed(0); env.reset(); obs=env.set_init_state(init)
   for _ in range(10): obs,_,_,_=env.step([0,0,0,0,0,0,-1])
   for step in range(target_step): obs,_,_,_=env.step(actions[step])
   _,image=prepare_agentview(obs)
   clean=predict_action(model,processor,image,task.language,unnorm_key='libero_90_no_noops')
   guided,meta=predict_matched(model,processor,image,task.language,entity_mode='source_target_libero90',query_mode='instruction_only',attention_layers=(11,),attention_heads=(),destination_weight=0.0,lambda_scale=1.0,unnorm_key='libero_90_no_noops',position_mode='attention')
   inputs=processor(build_prompt(task.language),image).to(model.device,dtype=torch.bfloat16)
   attention,h,ameta=prompt_attention_and_features(model,processor,image,task.language,inputs=inputs,query_mode='instruction_only',attention_layers=(11,),attention_heads=(),destination_weight=0.0)
   tokens=[int(x) for x in meta['selected_token_ids']]
   env_clean=np.asarray(clean,dtype=float); env_guided=np.asarray(guided,dtype=float)
   row={'case_id':cid,'group':e['group'],'task':task.name,'init_state_id':iid,'replay_step':target_step,
        'entities':meta['entities'],'matched_cluster_ids':meta['kmeans_groups'],'m':meta['m_matched'],'selected_unique':len(set(tokens)),
        'clean_action':env_clean.tolist(),'guided_action':env_guided.tolist(),
        'translation_delta':np.asarray(env_guided[:3]-env_clean[:3]).tolist(),'rotation_delta':np.asarray(env_guided[3:6]-env_clean[3:6]).tolist(),'gripper_delta':float(env_guided[6]-env_clean[6]),
        'clean_token_ids':meta['positive_token_ids'],'negative_token_ids':meta['negative_token_ids'],'final_token_ids':meta['final_token_ids'],
        'guided_changed_dims':meta['guided_changed_dims'],'feature_perturbation_relative':meta['feature_perturbation_relative'],
        'attention_sha256':meta['attention_sha256'],'mask_components':mask_components(tokens)}
   existing = json.loads((root/'ACTION_REPLAY.json').read_text()) if (root/'ACTION_REPLAY.json').exists() else {}
   existing[cid] = row
   atomic_json(root/'ACTION_REPLAY.json', existing)
   panel=Image.new('RGB',(224*3,224),(255,255,255)); panel.paste(image.resize((224,224)),(0,0)); panel.paste(overlay_heatmap(image,attention),(224,0)); panel.paste(overlay_mask(image,tokens),(448,0)); dr=ImageDraw.Draw(panel); dr.text((3,3),f'{cid} step={target_step}',fill=(255,255,255)); dr.text((227,3),'L11 attention',fill=(255,255,255)); dr.text((451,3),'selected mask',fill=(255,255,255)); panel.save(outdir/f'{cid}__action_mask.png')
   rows.append(row); print(json.dumps(row),flush=True)
  finally:
   env.close()
  (root/'ACTION_REPLAY_SUMMARY.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
 if rows:
  with (root/'ACTION_REPLAY.csv').open('w',newline='') as f:
   fields=['case_id','group','task','init_state_id','replay_step','entities','matched_cluster_ids','m','selected_unique','clean_action','guided_action','translation_delta','rotation_delta','gripper_delta','clean_token_ids','negative_token_ids','final_token_ids','guided_changed_dims','feature_perturbation_relative','attention_sha256','mask_components']
   w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
   for r in rows: w.writerow({k:(json.dumps(r[k]) if isinstance(r[k],(list,dict)) else r[k]) for k in fields})
if __name__=='__main__': main()
