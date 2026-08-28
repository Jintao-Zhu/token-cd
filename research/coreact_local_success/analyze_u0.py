from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import binomtest,spearmanr
ARMS=('Strong','W1_plus','W1_minus','Nearest_manifold','Random_smooth')
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();o=a.artifact.resolve();eps=[json.loads(x.read_text()) for x in o.joinpath('u0').glob('*.json')]
 if len(eps)!=2500 or list(o.joinpath('invalid_units').glob('*.json')):raise RuntimeError('U0 incomplete/invalid')
 units=defaultdict(dict)
 for e in eps:units[e['unit_id']][e['arm']]=e
 if len(units)!=500 or any(set(x)!=set(ARMS) for x in units.values()):raise RuntimeError('unit pairing')
 snaps=defaultdict(list)
 for uid,g in units.items():snaps[g['Strong']['snapshot_id']].append(g)
 if len(snaps)!=100 or any(len(x)!=5 for x in snaps.values()):raise RuntimeError('snapshot clustering')
 state=[]
 for sid,gs in sorted(snaps.items()):
  row={'snapshot_id':sid,'task_id':gs[0]['Strong']['task_id'],'progress':gs[0]['Strong']['progress']}
  for arm in ARMS:row[f'P_{arm}']=float(np.mean([g[arm]['success'] for g in gs]));row[f'U_{arm}']=row[f'P_{arm}']-row['P_Strong'] if arm!='Strong' else 0.
  state.append(row)
 rng=np.random.default_rng(20260817);summaries={};task_rows=[]
 for task in range(10):
  ss=[x for x in state if x['task_id']==task];task_rows.append({'task':task,**{arm:float(np.mean([x[f'P_{arm}'] for x in ss])) for arm in ARMS}})
 for arm in ARMS[1:]:
  diffs=np.asarray([x[f'U_{arm}'] for x in state]);boot=np.asarray([diffs[rng.integers(0,100,100)].mean()*100 for _ in range(10000)]);flat=np.asarray([int(g[arm]['success'])-int(g['Strong']['success']) for gs in snaps.values() for g in gs]);rescue=int((flat==1).sum());harm=int((flat==-1).sum());n=rescue+harm;nonworse=sum(x[arm]>=x['Strong'] for x in task_rows);summaries[arm]={'success':float(np.mean([x[f'P_{arm}'] for x in state])),'delta_pp':float(diffs.mean()*100),'snapshot_cluster_ci95_pp':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],'rescue':rescue,'harm':harm,'mcnemar_p':float(binomtest(min(rescue,harm),n).pvalue) if n else 1.,'tasks_nonworse':nonworse,'passes_fixed_direction_gate':bool(diffs.mean()*100>=3 and np.quantile(boot,.025)>0 and rescue>harm and nonworse>=7)}
 winners=[a for a in ARMS[1:] if summaries[a]['passes_fixed_direction_gate']];decision='FIXED_LOCAL_SUCCESS_DIRECTION_FOUND' if winners else 'U0_NO_FIXED_WINNER_TRIGGER_U1';result={'decision':decision,'winners':winners,'strong_success':float(np.mean([x['P_Strong'] for x in state])),'directions':summaries,'task_rows':task_rows,'state_utility_rows':state,'snapshots':100,'seeds_per_snapshot':5,'episodes':2500};(o/'u0_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
 with (o/'u0_state_utility.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=state[0].keys());w.writeheader();w.writerows(state)
 with (o/'u0_task_success.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=task_rows[0].keys());w.writeheader();w.writerows(task_rows)
 (o/'u0_decision.json').write_text(json.dumps({'decision':decision,'winners':winners},indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
