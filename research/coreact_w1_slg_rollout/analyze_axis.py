from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

ARMS=("lambda_n010","lambda_n005","lambda_0","lambda_005","lambda_010")
DOSES={"lambda_n010":-.10,"lambda_n005":-.05,"lambda_0":0.,"lambda_005":.05,"lambda_010":.10}
def comparison(rows,left,right,rng):
 d=np.asarray([int(x[left])-int(x[right]) for x in rows]);boot=np.asarray([d[rng.integers(0,500,500)].mean()*100 for _ in range(10000)]);r=int((d==1).sum());h=int((d==-1).sum());n=r+h
 return {'delta_pp':float(d.mean()*100),'ci95_pp':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],'rescue':r,'harm':h,'mcnemar_p':float(binomtest(min(r,h),n).pvalue) if n else 1.}
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();o=a.artifact.resolve();eps=[json.loads(x.read_text()) for x in o.joinpath('episodes').glob('*.json')]
 if len(eps)!=2500 or list(o.joinpath('invalid_pairs').glob('*.json')):raise RuntimeError('requires 2500 valid episodes')
 groups={}
 for e in eps:groups.setdefault(e['pair_id'],{})[e['arm']]=e
 if len(groups)!=500 or any(set(x)!=set(ARMS) for x in groups.values()):raise RuntimeError('pairing')
 rows=[{'pair':p,'task':g['lambda_0']['task_id'],**{arm:bool(g[arm]['success']) for arm in ARMS}} for p,g in sorted(groups.items())];rates={a:sum(x[a] for x in rows)/500 for a in ARMS};rng=np.random.default_rng(20260817);comparisons={a:comparison(rows,a,'lambda_0',rng) for a in ARMS if a!='lambda_0'}
 near=np.asarray([int(x['lambda_0'])-(int(x['lambda_n005'])+int(x['lambda_005']))/2 for x in rows]);far=np.asarray([int(x['lambda_0'])-(int(x['lambda_n010'])+int(x['lambda_010']))/2 for x in rows]);boot_near=np.asarray([near[rng.integers(0,500,500)].mean()*100 for _ in range(10000)]);boot_far=np.asarray([far[rng.integers(0,500,500)].mean()*100 for _ in range(10000)])
 tasks=[]
 for t in range(10):
  s=[x for x in rows if x['task']==t];r={'task':t,**{a:sum(x[a] for x in s) for a in ARMS}};r['near_local_peak_pp']=(r['lambda_0']-(r['lambda_n005']+r['lambda_005'])/2)*2;tasks.append(r)
 tasks_peak=sum(x['lambda_0']>=(x['lambda_n005']+x['lambda_005'])/2 for x in tasks);local={'local_peak_005_pp':float(near.mean()*100),'ci95_pp':[float(np.quantile(boot_near,.025)),float(np.quantile(boot_near,.975))],'tasks_strong_ge_neighbor_mean':tasks_peak};far_stat={'local_peak_010_pp':float(far.mean()*100),'ci95_pp':[float(np.quantile(boot_far,.025)),float(np.quantile(boot_far,.975))]}
 confirmed=rates['lambda_0']>rates['lambda_n005'] and rates['lambda_0']>rates['lambda_005'] and local['local_peak_005_pp']>=2 and local['ci95_pp'][0]>0 and tasks_peak>=7
 if confirmed:decision='W1_RESIDUAL_AXIS_LOCAL_OPTIMUM_CONFIRMED'
 elif any(comparisons[a]['delta_pp']>=2 and comparisons[a]['ci95_pp'][0]>0 and comparisons[a]['rescue']>comparisons[a]['harm'] for a in ('lambda_n010','lambda_n005')):decision='W1_AXIS_PREFERS_SHRINKAGE'
 elif any(comparisons[a]['delta_pp']>=2 and comparisons[a]['ci95_pp'][0]>0 and comparisons[a]['rescue']>comparisons[a]['harm'] for a in ('lambda_005','lambda_010')):decision='W1_AXIS_PREFERS_EXTRAPOLATION'
 else:decision='W1_AXIS_LOCAL_OPTIMUM_INCONCLUSIVE'
 metrics={}
 for arm in ARMS:
  ts=[t for e in eps if e['arm']==arm for t in e['replan_traces']];clips=[c for t in ts for c in t['clip_scales']];corr=[c for t in ts for c in t['velocity_correction_norms']];metrics[arm]={'action_delta_l2':float(np.mean([t['normalized_action_delta_l2'] for t in ts])),'xyz_delta':float(np.mean([t['normalized_action_delta_xyz'] for t in ts])),'rotation_delta':float(np.mean([t['normalized_action_delta_rotation'] for t in ts])),'gripper_delta':float(np.mean([t['normalized_action_delta_gripper'] for t in ts])),'velocity_correction_norm':float(np.mean(corr)) if corr else 0.,'clip_rate':float(np.mean(np.asarray(clips)<1)) if clips else 0.}
 symmetry={'action_delta_005_ratio_abs':metrics['lambda_n005']['action_delta_l2']/metrics['lambda_005']['action_delta_l2'],'action_delta_010_ratio_abs':metrics['lambda_n010']['action_delta_l2']/metrics['lambda_010']['action_delta_l2'],'symmetric_enough':abs(metrics['lambda_n005']['action_delta_l2']/metrics['lambda_005']['action_delta_l2']-1)<.1 and abs(metrics['lambda_n010']['action_delta_l2']/metrics['lambda_010']['action_delta_l2']-1)<.1}
 result={'decision':decision,'rates':rates,'comparisons':comparisons,'primary_local_peak':local,'secondary_local_peak':far_stat,'residual_metrics':metrics,'symmetry_audit':symmetry,'tasks':tasks,'episodes':2500,'pairs':500};(o/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');(o/'decision.json').write_text(json.dumps({'decision':decision},indent=2)+'\n');(o/'status.json').write_text(json.dumps({'status':'complete','episodes_complete':2500,'invalid_pairs':0,'decision':decision},indent=2)+'\n')
 with (o/'task_success.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=tasks[0].keys());w.writeheader();w.writerows(tasks)
 plt.figure(figsize=(6,4));xs=[DOSES[a] for a in ARMS];ys=[rates[a]*100 for a in ARMS];plt.plot(xs,ys,marker='o');plt.axvline(0,color='gray',lw=1);plt.xlabel('Residual-axis lambda');plt.ylabel('Success (%)');plt.grid(alpha=.25);plt.tight_layout();plt.savefig(o/'residual_axis_curve.png',dpi=180);plt.close();print(json.dumps(result,indent=2))
if __name__=='__main__':main()
