from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, binomtest
from .common import file_sha256, read_jsonl, write_json

CONDS=("vanilla","top16_mask_only","away","toward")
def mcnemar(a,b):
 x=int(np.sum((a==1)&(b==0))); y=int(np.sum((a==0)&(b==1))); n=x+y
 return {'first_only':x,'second_only':y,'discordant':n,'exact_p':1.0 if n==0 else float(binomtest(min(x,y),n=n,p=0.5).pvalue)}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--artifact',type=Path,required=True); a=p.parse_args(); art=a.artifact.resolve(); manifest=read_jsonl(art/'rollout_manifest.lock.jsonl'); files=list((art/'episodes').glob('*.json')); rows=[json.loads(x.read_text()) for x in files]
 expected={x['episode_id'] for x in manifest}; actual={x['episode_id'] for x in rows}; by={}
 for x in rows: by.setdefault(x['snapshot_id'],{})[x['condition']]=x
 checks={'episodes_200':len(rows)==200,'manifest_200':len(manifest)==200,'ids_exact':actual==expected,'ids_unique':len(actual)==len(rows),'paired_50':len(by)==50 and all(set(v)==set(CONDS) for v in by.values()),'all_replans_masked':all(x['all_replans_masked'] for x in rows),'finite':all(x['all_actions_finite'] for x in rows)}
 if not all(checks.values()): raise RuntimeError(json.dumps(checks))
 snapshots=sorted(by); paired=[]
 for sid in snapshots:
  v=by[sid]['vanilla']
  for c in CONDS[1:]:
   t=by[sid][c]; paired.append({'snapshot_id':sid,'task_id':v['task_id'],'condition':c,'vanilla_success':int(v['success']),'treated_success':int(t['success']),'success_discordance':int(v['success']!=t['success']),'object_goal_abs_effect':abs(t['final_progress']['object_goal_distance']-v['final_progress']['object_goal_distance']),'eef_object_abs_effect':abs(t['final_progress']['eef_object_distance']-v['final_progress']['eef_object_distance']),'offline_effect':float(t.get('offline_effect',0.0))})
 with (art/'paired_results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(paired[0])); w.writeheader(); w.writerows(paired)
 rates={c:float(np.mean([by[s][c]['success'] for s in snapshots])) for c in CONDS}; metrics={c:{k:float(np.mean([r[k] for r in paired if r['condition']==c])) for k in ('success_discordance','object_goal_abs_effect','eef_object_abs_effect')} for c in CONDS[1:]}
 rng=np.random.default_rng(20260809); task_ids=np.array([by[s]['vanilla']['task_id'] for s in snapshots]); boots={k:[] for k in ('mask_minus_away','mask_minus_toward','rho')}
 for _ in range(2000):
  idx=[]
  for task in rng.choice(np.unique(task_ids),size=3,replace=True):
   pool=np.flatnonzero(task_ids==task); idx.extend(rng.choice(pool,size=len(pool),replace=True))
  def mean(c,k): return float(np.mean([ [r for r in paired if r['snapshot_id']==snapshots[i] and r['condition']==c][0][k] for i in idx]))
  boots['mask_minus_away'].append(mean('top16_mask_only','object_goal_abs_effect')-mean('away','object_goal_abs_effect'))
  boots['mask_minus_toward'].append(mean('top16_mask_only','object_goal_abs_effect')-mean('toward','object_goal_abs_effect'))
  flat=[r for i in idx for r in paired if r['snapshot_id']==snapshots[i]]; boots['rho'].append(float(spearmanr([r['offline_effect'] for r in flat],[r['object_goal_abs_effect'] for r in flat]).statistic))
 ci=lambda x:[float(np.quantile(x,.025)),float(np.quantile(x,.975))]
 comps={'top16_minus_away_object_goal':{'point':metrics['top16_mask_only']['object_goal_abs_effect']-metrics['away']['object_goal_abs_effect'],'ci95':ci(boots['mask_minus_away'])},'top16_minus_toward_object_goal':{'point':metrics['top16_mask_only']['object_goal_abs_effect']-metrics['toward']['object_goal_abs_effect'],'ci95':ci(boots['mask_minus_toward'])},'offline_vs_object_goal_spearman':{'point':float(spearmanr([r['offline_effect'] for r in paired],[r['object_goal_abs_effect'] for r in paired]).statistic),'ci95':ci(boots['rho'])}}
 mc={c:mcnemar(np.array([by[s]['vanilla']['success'] for s in snapshots]),np.array([by[s][c]['success'] for s in snapshots])) for c in CONDS[1:]}
 summary={'completion':{'episodes':len(rows),'paired_snapshots':len(by),'missing':0,'duplicates':0},'success_rates':rates,'magnitude_metrics':metrics,'comparisons':comps,'mcnemar_vs_vanilla':mc,'integrity':checks,'decision':'AR_EVERY_REPLAN_DIRECTION_REPLICATION_COMPLETE'}
 write_json(art/'summary.json',summary); write_json(art/'decision.json',{'decision':summary['decision'],'integrity':checks});
 (art/'report.md').write_text('# AR Every-Replan Direction Replication\n\n'+json.dumps(summary,indent=2,sort_keys=True)+'\n\nThis is a development replication. Away/toward are action-space extrapolations with scale 0.5; no causal sign claim is made beyond this locked experiment.\n',encoding='utf-8')
 write_json(art/'sha256_audit.json',{str(x.relative_to(art)):file_sha256(x) for x in sorted(art.rglob('*')) if x.is_file() and x.name!='sha256_audit.json'}); print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__': main()
