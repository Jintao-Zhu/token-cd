"""Score the frozen SpatialVLA all-layer offline attention screen."""
from __future__ import annotations
import csv,json
from itertools import combinations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[2]
OUT=REPO/'artifacts/spatialvla_layer_selection_v1b'
LAYERS=tuple(range(26))
GROUP_TASKS={
 'fractal':('google_robot_open_drawer','google_robot_close_drawer','google_robot_pick_coke_can','google_robot_move_near','google_robot_place_apple_in_closed_top_drawer'),
 'bridge':('widowx_carrot_on_plate','widowx_put_eggplant_in_basket','widowx_spoon_on_towel','widowx_stack_cube'),
}
def topm(s,m): return np.lexsort((np.arange(len(s)),-s))[:m]
def jac(a,b):
 x,y=set(map(int,a)),set(map(int,b)); return len(x&y)/max(1,len(x|y))
def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def percentile(values,value):
 a=np.asarray(values); return float(((a<value).sum()+.5*(a==value).sum())/len(a))
def load(group,split):
 rows=[]
 for task in GROUP_TASKS[group]:
  for seed in (100,110,120,130,150,160):
   d=OUT/'states'/task/f'seed_{seed:03d}'; meta=json.loads((d/'layers.json').read_text())
   this_split='exploration' if seed in (100,110,120,130) else 'validation'
   if this_split!=split: continue
   a=dict(np.load(d/'layers.npz'))
   rows.append((meta,a))
 return rows
def eval_cfg(rows,layers):
 syn_j=[];syn_cos=[]; controls=[];by_task={}
 for meta,a in rows:
  l=np.asarray(layers,dtype=int); orig=a['original'][l].mean(0); syn=a['synonym'][l].mean(0); target=a['target_switch'][l].mean(0);m=int(meta['m'])
  om=topm(orig,m);sm=topm(syn,m);tm=topm(target,m)
  syn_j.append(jac(om,sm));syn_cos.append(cos(orig,syn))
  old=np.flatnonzero(a['old_target_mask']);new=np.flatnonzero(a['new_target_mask'])
  on=orig/max(1e-12,float(orig.sum()));tn=target/max(1e-12,float(target.sum()))
  new_gain=float(tn[new].sum()-on[new].sum());old_drop=float(on[old].sum()-tn[old].sum())
  new_mask_gain=(len(set(tm)&set(new))-len(set(om)&set(new)))/m
  old_mask_drop=(len(set(om)&set(old))-len(set(tm)&set(old)))/m
  response=.5*(new_gain+old_drop); mask_response=.5*(new_mask_gain+old_mask_drop)
  row={'task':meta['task'],'seed':meta['seed'],'attention_response':response,'mask_response':mask_response,'correct_both':bool(new_gain>0 and old_drop>0),'new_target_gain':new_gain,'old_target_drop':old_drop,'target_mask_jaccard':jac(om,tm),'m':m}
  controls.append(row);by_task.setdefault(meta['task'],[]).append(response)
 return {'layers':list(map(int,layers)),'target_response':float(np.mean([r['attention_response'] for r in controls])),'target_mask_response':float(np.mean([r['mask_response'] for r in controls])),'correct_response_fraction':float(np.mean([r['correct_both'] for r in controls])),'positive_task_count':sum(float(np.mean(v))>0 for v in by_task.values()),'target_response_by_task':{k:float(np.mean(v)) for k,v in by_task.items()},'synonym_mask_jaccard':float(np.mean(syn_j)),'synonym_score_cosine':float(np.mean(syn_cos)),'target_controls':controls}
def score(rows):
 singles=[eval_cfg(rows,(i,)) for i in LAYERS]
 tv=[r['target_response'] for r in singles];mv=[r['target_mask_response'] for r in singles];cv=[r['correct_response_fraction'] for r in singles];sv=[r['synonym_mask_jaccard'] for r in singles]
 for r in singles:r['selection_score']=.55*percentile(tv,r['target_response'])+.20*percentile(mv,r['target_mask_response'])+.15*percentile(cv,r['correct_response_fraction'])+.10*percentile(sv,r['synonym_mask_jaccard'])
 eligible=[r for r in singles if r['positive_task_count']>=2]; ranked=sorted(eligible or singles,key=lambda r:(-r['selection_score'],r['layers'][0]))
 shortlist=[r['layers'][0] for r in ranked[:4]]
 pairs=[eval_cfg(rows,pair) for pair in combinations(shortlist,2)]
 for r in pairs:r['selection_score']=.55*percentile(tv,r['target_response'])+.20*percentile(mv,r['target_mask_response'])+.15*percentile(cv,r['correct_response_fraction'])+.10*percentile(sv,r['synonym_mask_jaccard'])
 bestpair=max(pairs,key=lambda r:(r['selection_score'],-sum(r['layers'])))['layers'] if pairs else None
 return singles,ranked,shortlist,pairs,bestpair
def save_csv(path,rows):
 fields=('layers','selection_score','target_response','target_mask_response','correct_response_fraction','positive_task_count','synonym_mask_jaccard','synonym_score_cosine')
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(r[k]) if k=='layers' else r.get(k) for k in fields})
def main():
 for group in GROUP_TASKS:
  exp=load(group,'exploration'); val=load(group,'validation')
  singles,ranked,shortlist,pairs,bestpair=score(exp)
  lock={'protocol_id':'SPATIALVLA_LAYER_SELECTION_OFFLINE_V1','checkpoint_group':group,'selection_data':f"exploration only ({len(exp)} states)",'validation_was_read':False,'num_text_layers':26,'candidate_single_layers':shortlist,'prompt_single':[shortlist[0]],'prompt_sparse':bestpair,'legacy_l11':[11],'tested_pairs':[r['layers'] for r in pairs],'selection_rule':'0.55 target attention response percentile + 0.20 target-mask response + 0.15 correct-both fraction + 0.10 synonym Top-m stability; prefer layers with positive mean response on at least two tasks when possible','single_metrics':ranked[0],'sparse_metrics':next((r for r in pairs if r['layers']==bestpair),None)}
  gp=OUT/group;gp.mkdir(parents=True,exist_ok=True)
  cp=gp/'CANDIDATES_LOCK.json'
  if cp.exists() and json.loads(cp.read_text())!=lock: raise RuntimeError(f'candidate lock mismatch {group}')
  cp.write_text(json.dumps(lock,indent=2,sort_keys=True)+'\n')
  save_csv(gp/'exploration_per_layer.csv',singles);save_csv(gp/'exploration_candidate_pairs.csv',pairs)
  x=np.arange(26);fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
  axes[0].plot(x,[r['target_response'] for r in singles],marker='o');axes[0].axhline(0,color='black',lw=.7);axes[0].set_ylabel('target response')
  axes[1].plot(x,[r['correct_response_fraction'] for r in singles],marker='o',label='correct both');axes[1].plot(x,[r['synonym_mask_jaccard'] for r in singles],marker='o',label='synonym Top-m Jaccard');axes[1].legend();axes[1].set_ylabel('fraction / Jaccard')
  axes[2].plot(x,[r['selection_score'] for r in singles],marker='o');axes[2].set_ylabel('selection score');axes[2].set_xlabel('SpatialVLA text layer (0-based)')
  for ax in axes:
   for layer in shortlist:ax.axvline(layer,color='tab:red',alpha=.18)
   ax.grid(alpha=.2)
  fig.suptitle(f'SpatialVLA layer screen — {group}, exploration only');fig.tight_layout();fig.savefig(gp/'exploration_layer_curve.png',dpi=180);plt.close(fig)
  # The candidate set is now frozen. Only now evaluate held-out states.
  configs={f'candidate_L{i}':(i,) for i in shortlist};configs['legacy_L11']=(11,)
  if bestpair:configs['best_pair']=tuple(bestpair)
  vr={name:eval_cfg(val,layers) for name,layers in configs.items()}
  result={'protocol_id':'SPATIALVLA_LAYER_SELECTION_OFFLINE_V1','checkpoint_group':group,'selection_lock_sha256':__import__('hashlib').sha256(cp.read_bytes()).hexdigest(),'validation_data':f"held-out states ({len(val)}); evaluated after candidate lock",'results':vr}
  (gp/'VALIDATION_LAYER_RESULTS.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
  summary={'checkpoint_group':group,'exploration_states':len(exp),'validation_states':len(val),'top_single_layers':[(r['layers'][0],round(r['selection_score'],4),round(r['target_response'],4),round(r['synonym_mask_jaccard'],3),r['positive_task_count']) for r in ranked[:8]],'chosen_single':shortlist[0],'chosen_pair':bestpair,'validation':{k:{'layers':v['layers'],'target_response':round(v['target_response'],4),'target_mask_response':round(v['target_mask_response'],4),'correct_response_fraction':round(v['correct_response_fraction'],3),'synonym_mask_jaccard':round(v['synonym_mask_jaccard'],3),'task_response':{t:round(x,4) for t,x in v['target_response_by_task'].items()}} for k,v in vr.items()}}
  (gp/'SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
  print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
