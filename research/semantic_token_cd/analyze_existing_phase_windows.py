#!/usr/bin/env python3
from __future__ import annotations
import csv,json,collections
from pathlib import Path
import numpy as np
try:
 from scipy.stats import mannwhitneyu
except Exception: mannwhitneyu=None
root=Path('artifacts/libero90_five_task_simpler_config_v1')
states=list(csv.DictReader(open(root/'STATE_DIAGNOSTICS.csv')))
rep={(r['case_id'],r['arm']):r for r in csv.DictReader(open(root/'REPLAY_DIAGNOSTICS.csv'))}
events=list(csv.DictReader(open(root/'CASE_EVENT_TABLE.csv')))
bycase=collections.defaultdict(list)
for s in states: bycase[s['case_id']].append(s)
for arr in bycase.values(): arr.sort(key=lambda x:int(x['step']))
rows=[]
for e in events:
 rec=rep.get((e['case_id'],'matched'),{})
 gs=rec.get('first_target_grasp_step','')
 if gs in ('',None): continue
 g=int(gs); arr=bycase[e['case_id']]
 pre=[x for x in arr if max(0,g-10)<=int(x['step'])<g]
 post=[x for x in arr if g<=int(x['step'])<min(len(arr),g+10)]
 def avg(xs,k): return float(np.mean([float(x[k]) for x in xs])) if xs else None
 row={'case_id':e['case_id'],'group':e['group'],'task':e['task'],'init_state_id':e['init_state_id'],'first_target_grasp_step':g,
      'pre_change_fraction':float(np.mean([int(float(x['guided_changed_dims']))>0 for x in pre])) if pre else None,
      'post_change_fraction':float(np.mean([int(float(x['guided_changed_dims']))>0 for x in post])) if post else None,
      'pre_mean_changed_dims':avg(pre,'guided_changed_dims'),'post_mean_changed_dims':avg(post,'guided_changed_dims'),
      'pre_mean_perturbation':avg(pre,'feature_perturbation_relative'),'post_mean_perturbation':avg(post,'feature_perturbation_relative'),
      'pre_mean_m':avg(pre,'m_effective'),'post_mean_m':avg(post,'m_effective'),
      'pre_mean_budget_jump':avg(pre,'budget_jump'),'post_mean_budget_jump':avg(post,'budget_jump'),
      'pre_steps':len(pre),'post_steps':len(post)}
 rows.append(row)
with (root/'PHASE_WINDOW_SUMMARY.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
summary={}
for group in ('rescue','harm','both_success','both_fail'):
 rr=[r for r in rows if r['group']==group]
 summary[group]={}
 for k in ['pre_change_fraction','post_change_fraction','pre_mean_changed_dims','post_mean_changed_dims','pre_mean_perturbation','post_mean_perturbation','pre_mean_m','post_mean_m','pre_mean_budget_jump','post_mean_budget_jump']:
  v=[r[k] for r in rr if r[k] is not None]
  summary[group][k]=float(np.mean(v)) if v else None
for k in ['pre_mean_changed_dims','post_mean_changed_dims','pre_change_fraction','post_change_fraction','pre_mean_perturbation','post_mean_perturbation']:
 r=[x[k] for x in rows if x['group']=='rescue' and x[k] is not None]; h=[x[k] for x in rows if x['group']=='harm' and x[k] is not None]
 if mannwhitneyu and r and h: print(k,'rescue',np.mean(r),'harm',np.mean(h),'p',mannwhitneyu(r,h,alternative='two-sided').pvalue)
(root/'PHASE_WINDOW_SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps(summary,indent=2))
