"""Formal-only paired factorial analysis. No test-outcome parameter selection.
Predeclared families: ranking3, budget4, structure2; interaction exploratory2.
Intervals marginal95%, not simultaneous. Resample shared seeds within each task.
"""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.stats import binomtest

def holm(p):
 order=np.argsort(p); out=np.zeros(len(p)); prev=0
 for rank,i in enumerate(order):
  prev=max(prev,min(1.,p[i]*(len(p)-rank))); out[i]=prev
 return out.tolist()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--artifact',type=Path,required=True);ap.add_argument('--bootstrap',type=int,default=20000);a=ap.parse_args()
 c=json.loads((a.artifact/'CONFIG_LOCK.json').read_text());audit=json.loads((a.artifact/'FORMAL_MEMBERSHIP_AUDIT.json').read_text())
 if not audit['passed']:raise RuntimeError('formal audit must pass')
 arms=c['arms'];tasks=c['tasks'];seeds=c['formal_seeds'];cube=np.empty((len(tasks),len(seeds),len(arms)),dtype=int)
 records=[]
 for ti,t in enumerate(tasks):
  for si,s in enumerate(seeds):
   for ai,arm in enumerate(arms):
    d=json.loads((a.artifact/'episodes/formal'/t/arm/f'episode_{s:03d}_summary.json').read_text())
    if not d['formal_eligible'] or not d['technical_pass']:raise RuntimeError('ineligible episode')
    cube[ti,si,ai]=int(d['success'])
 def diff(left,right):return cube[:,:,arms.index(left)]-cube[:,:,arms.index(right)]
 rng=np.random.default_rng(4823)
 # Same bootstrap indices across every contrast, stratified by task.
 idx=rng.integers(len(seeds),size=(a.bootstrap,len(tasks),len(seeds)))
 def boot_ci(d):
  vals=np.stack([d[t][idx[:,t,:]].mean(1) for t in range(len(tasks))],axis=1).mean(1)*100
  return np.percentile(vals,[2.5,97.5]).tolist()
 pooled_rng=np.random.default_rng(7761); pooled_idx=pooled_rng.integers(len(tasks)*len(seeds),size=(a.bootstrap,len(tasks)*len(seeds)))
 def boot_ci_pooled(d):
  flat=d.reshape(len(tasks)*len(seeds))
  vals=flat[pooled_idx].mean(1)*100
  return np.percentile(vals,[2.5,97.5]).tolist()
 families={'ranking':[(f'attn_{b}',f'kmeans_{b}') for b in ['matched','fixed','topp']],
 'budget':[(f'{r}_matched',f'{r}_{b}') for r in ['attn','kmeans'] for b in ['fixed','topp']],
 'structure':[('attn_matched','random_matched'),('attn_matched','norm_noise')]}
 output={'protocol':c['protocol_id'],'n_per_task':len(seeds),'tasks':tasks,'interval':'marginal paired task-stratified percentile bootstrap95%; not simultaneous',
 'arms':{arm:{'success':int(cube[:,:,i].sum()),'n':len(tasks)*len(seeds),'per_task':{t:int(cube[j,:,i].sum()) for j,t in enumerate(tasks)}} for i,arm in enumerate(arms)},'families':{}}
 for fam,pairs in families.items():
  rows=[]
  for left,right in pairs:
   d=diff(left,right);b=int((d==1).sum());h=int((d==-1).sum());p=binomtest(b,b+h,.5).pvalue if b+h else 1.
   rows.append({'left':left,'right':right,'diff_pp':float(d.mean()*100),'ci95_pp':boot_ci(d),
                'ci95_pp_pooled_resample':boot_ci_pooled(d),
                'weighting':'equal task weighting; every task has the same number of seeds so the pooled point estimate is identical',
                'left_only':b,'right_only':h,'p_exact':p,
                'per_task_diff_pp':{t:float(d[i].mean()*100) for i,t in enumerate(tasks)}})
  for row,p in zip(rows,holm([r['p_exact'] for r in rows])):row['p_holm']=p
  output['families'][fam]=rows
 output['exploratory_interactions']=[]
 for b in ['fixed','topp']:
  d=diff('attn_matched',f'attn_{b}')-diff('kmeans_matched',f'kmeans_{b}')
  output['exploratory_interactions'].append({'budget_comparison':'matched-minus-'+b,'difference_of_differences_pp':float(d.mean()*100),'ci95_pp':boot_ci(d),'interpretation':'Exploratory interaction; do not infer from separate significance tests'})
 output['vanilla_descriptive']=[{'arm':arm,'diff_pp':float(diff(arm,'vanilla').mean()*100),'ci95_pp':boot_ci(diff(arm,'vanilla'))} for arm in arms if arm!='vanilla']
 p=a.artifact/'FORMAL_ANALYSIS.json'
 if p.exists():raise RuntimeError('analysis already exists; review before replacement')
 p.write_text(json.dumps(output,indent=2)+'\n'); print(p)
if __name__=='__main__':main()
