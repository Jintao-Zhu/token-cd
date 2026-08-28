from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();o=a.artifact.resolve();eps=[json.loads(x.read_text()) for x in o.joinpath('episodes').glob('*.json')]
 if len(eps)!=500 or list(o.joinpath('invalid_pairs').glob('*.json')): raise RuntimeError('requires 500 valid episodes')
 g={}
 for e in eps:g.setdefault(e['pair_id'],{})[e['arm']]=e
 if len(g)!=250 or any(set(x)!=set(('lambda_0','lambda_001')) for x in g.values()):raise RuntimeError('pairing')
 rows=[]
 for pair,x in sorted(g.items()):rows.append({'pair':pair,'task':x['lambda_0']['task_id'],'strong':bool(x['lambda_0']['success']),'dose':bool(x['lambda_001']['success'])})
 d=np.asarray([int(x['dose'])-int(x['strong']) for x in rows]);rng=np.random.default_rng(20260817);boot=np.asarray([d[rng.integers(0,250,250)].mean()*100 for _ in range(10000)]);rescue=int((d==1).sum());harm=int((d==-1).sum());n=rescue+harm;sr0=sum(x['strong'] for x in rows)/250;sr1=sum(x['dose'] for x in rows)/250
 traces=[t for e in eps if e['arm']=='lambda_001' for t in e['replan_traces']];delta=float(np.mean([t['normalized_action_delta_l2'] for t in traces]));clips=[c for t in traces for c in t['clip_scales']];clip=float(np.mean(np.asarray(clips)<1))
 tasks=[{'task':t,'lambda_0':sum(x['strong'] for x in rows if x['task']==t),'lambda_001':sum(x['dose'] for x in rows if x['task']==t)} for t in range(10)];result={'decision':'W1_LAMBDA_001_EXPLORATORY_GAIN' if sr1>sr0 else 'W1_LAMBDA_001_EXPLORATORY_NO_GAIN','exploratory_posthoc':True,'strong_success':sr0,'lambda_001_success':sr1,'delta_pp':float((sr1-sr0)*100),'ci95_pp':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],'rescue':rescue,'harm':harm,'mcnemar_p':float(binomtest(min(rescue,harm),n).pvalue) if n else 1.,'mean_action_delta_l2':delta,'clip_rate':clip,'tasks':tasks}
 (o/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');(o/'decision.json').write_text(json.dumps({'decision':result['decision'],'exploratory_posthoc':True},indent=2)+'\n');(o/'status.json').write_text(json.dumps({'status':'complete','episodes_complete':500,'decision':result['decision'],'invalid_pairs':0},indent=2)+'\n')
 plt.figure(figsize=(4.5,3.8));plt.plot([0,.01],[sr0*100,sr1*100],marker='o');plt.xticks([0,.01],['0','.01']);plt.ylabel('Success (%)');plt.xlabel('lambda');plt.grid(alpha=.25);plt.tight_layout();plt.savefig(o/'lambda001_curve.png',dpi=180);plt.close()
 task='\n'.join(f"| {x['task']} | {x['lambda_0']}/25 | {x['lambda_001']}/25 |" for x in tasks);report=f"""# Exploratory W1 lambda=.01 Closed-Loop Test

`{result['decision']}`

| lambda | Success | Delta | Rescue | Harm | Action delta | Clip rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | {sum(x['strong'] for x in rows)}/250 ({sr0:.1%}) | - | - | - | 0 | 0% |
| .01 | {sum(x['dose'] for x in rows)}/250 ({sr1:.1%}) | {(sr1-sr0)*100:+.1f}pp | {rescue} | {harm} | {delta:.4f} | {clip:.1%} |

Paired bootstrap 95% CI: [{result['ci95_pp'][0]:+.1f}, {result['ci95_pp'][1]:+.1f}]pp. Exact McNemar p={result['mcnemar_p']:.6g}.

| Task | lambda=0 | lambda=.01 |
|---:|---:|---:|
{task}

This is a post-hoc exploratory extension after the preregistered dose-response NO-GO. It is not an independent confirmation.
""";(o/'REPORT.md').write_text(report);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
