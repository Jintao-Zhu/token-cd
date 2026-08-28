#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np

def boot(rows,key,n=10000,seed=20260818):
 by=defaultdict(list)
 for r in rows:by[r['task_id']].append(r[key])
 rng=np.random.default_rng(seed);d=np.empty(n);tasks=sorted(by)
 for i in range(n):d[i]=np.mean(np.concatenate([rng.choice(by[t],len(by[t]),replace=True) for t in tasks]))
 return [float(np.quantile(d,.025)),float(np.quantile(d,.975))]
def aggregate(records,steps):
 out=[]
 for rec in records:
  rs=[r for r in rec['rows'] if r['flow_step'] in steps];out.append({'task_id':rec['state']['task_id'],'state_id':rec['state']['state_id'],**{k:float(np.mean([r[k] for r in rs])) for k in ('marginal_g_positive','nearest_g_positive','delta_d_applied_positive','weak_to_strong_manifold_distance_ratio','weak_to_strong_marginal_error_ratio','direction_relative_norm','clip_scale')}})
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();art=a.artifact.resolve();records=[json.loads(x.read_text()) for x in sorted((art/'confirmation').glob('*.json'))]
 if len(records)!=250 or any(not r['integrity']['finite'] or r['integrity']['rows']!=12 or r['integrity']['all_ones_strong_parity_max_abs']>=1e-6 for r in records):raise RuntimeError(f'invalid confirmation {len(records)}')
 low=aggregate(records,[8,9]);high=aggregate(records,[0,1]);result={}
 for name,rows in [('low_noise',low),('high_noise_placebo',high)]:
  result[name]={'states':len(rows)}
  for k in ('marginal_g_positive','nearest_g_positive','delta_d_applied_positive'):
   result[name][k]=float(np.mean([r[k] for r in rows]));result[name][k+'_ci95']=boot(rows,k)
  for k in ('weak_to_strong_manifold_distance_ratio','weak_to_strong_marginal_error_ratio','direction_relative_norm','clip_scale'):result[name][k]=float(np.mean([r[k] for r in rows]))
 task=[]
 for t in range(10):
  x=[r for r in low if r['task_id']==t];task.append({'task_id':t,'states':len(x),'marginal_g_positive':float(np.mean([r['marginal_g_positive'] for r in x])),'delta_d_applied_positive':float(np.mean([r['delta_d_applied_positive'] for r in x]))})
 checks={'marginal_ge_60':result['low_noise']['marginal_g_positive']>=.60,'marginal_ci_lower_gt_50':result['low_noise']['marginal_g_positive_ci95'][0]>.50,'delta_d_ge_55':result['low_noise']['delta_d_applied_positive']>=.55,'tasks_gt_50_ge_7':sum(r['marginal_g_positive']>.5 for r in task)>=7,'low_marginal_gt_high':result['low_noise']['marginal_g_positive']>result['high_noise_placebo']['marginal_g_positive'],'low_delta_d_gt_high':result['low_noise']['delta_d_applied_positive']>result['high_noise_placebo']['delta_d_applied_positive']}
 decision='SLG_SELF_WEAK_LOW_NOISE_COMPATIBILITY_CONFIRMED' if all(checks.values()) else 'SLG_SELF_WEAK_SELECTION_PASS_CONFIRMATION_FAIL';out={'status':'SELF_WEAK_CONFIRMATION_COMPLETE','decision':decision,'integrity':{'states':250,'rows_per_state':12,'normalization_max_abs':max(r['integrity']['normalization_max_abs'] for r in records),'strong_parity_max_abs':max(r['integrity']['all_ones_strong_parity_max_abs'] for r in records)},'low_noise':result['low_noise'],'high_noise_placebo':result['high_noise_placebo'],'tasks':task,'checks':checks};(art/'confirmation_result.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');(art/'status/current.json').write_text(json.dumps({'stage':'CONFIRMATION_COMPLETE','decision':decision},indent=2)+'\n');print(json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
