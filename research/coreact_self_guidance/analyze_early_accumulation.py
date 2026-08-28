#!/usr/bin/env python3
from __future__ import annotations
import csv,glob,hashlib,json,math,random
from collections import defaultdict
from pathlib import Path
ART=Path('artifacts/coreact_task4_early_trajectory_accumulation_v1_20260810_122858');ARMS=('V_vanilla','W2_full','W2_early25','W2_start25','W2_early40','W2_start40')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def comp(rows,a,b,seed):
 d=[r[a]-r[b] for r in rows];rng=random.Random(seed);z=sorted(sum(d[rng.randrange(len(d))] for _ in d)/len(d) for _ in range(2000));ao=sum(r[a] and not r[b] for r in rows);bo=sum(r[b] and not r[a] for r in rows);n=ao+bo;k=min(ao,bo);p=min(1,2*sum(math.comb(n,i) for i in range(k+1))*.5**n) if n else 1
 return {'comparison':f'{a}_minus_{b}','difference':sum(d)/len(d),'bootstrap_95_ci':[z[49],z[1949]],'first_only_success':ao,'second_only_success':bo,'exact_mcnemar_p':p}
def main():
 by=defaultdict(dict);bad=[];ph=sha(ART/'protocol.lock.yaml')
 for p in sorted((ART/'episodes').glob('*.json')):
  x=json.loads(p.read_text());by[x['pair_id']][x['arm']]=x;gp=ART/x['geometry_path']
  if x['status']!='complete' or not x['all_actions_finite'] or x['protocol_sha256']!=ph or not gp.is_file() or sha(gp)!=x['geometry_sha256']:bad.append(x['episode_id'])
 rows=[]
 for pair,a in sorted(by.items()):
  if set(a)!=set(ARMS):bad.append(pair);continue
  if len({json.dumps(a[x]['initial_gate'],sort_keys=True) for x in ARMS})!=1:bad.append(pair+' gate')
  rows.append({'pair_id':pair,**{x:int(a[x]['success']) for x in ARMS},'L':a['V_vanilla']['vanilla_reference_length'],'cutoff25':a['V_vanilla']['cutoff25'],'cutoff40':a['V_vanilla']['cutoff40']})
 with (ART/'paired_results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 rates={a:sum(r[a] for r in rows)/len(rows) for a in ARMS};comparisons=[comp(rows,a,'V_vanilla',20260810+i) for i,a in enumerate(ARMS[1:])];comparisons += [comp(rows,'W2_early25','W2_start25',20260901),comp(rows,'W2_early40','W2_start40',20260902)]
 integrity={'pass':not bad and len(rows)==50,'pairs':len(rows),'episodes':sum(len(x) for x in by.values()),'geometry':len(list((ART/'geometry').glob('*.pt'))),'invalid_pairs':len(list((ART/'invalid_pairs').glob('*.json'))),'bad':bad}
 decision={'status':'STOP_TIMESTEP_SG_PERFORMANCE_OPTIMIZATION_RETURN_TO_TOKEN_LINE','reason':'Fresh-seed full W2 improves by only 6pp with a paired 95% CI crossing zero; early-only arms improve by 2pp and early-versus-start differences are directional but uncertain. The prior +12pp effect is not stably reproduced, so cumulative trajectory shaping is not established.','selector_route':'NO_GO','trajectory_shaping_established':False}
 summary={'integrity':integrity,'success_rates':rates,'comparisons':comparisons,'prior_context':{'vanilla':.54,'W2_full':.66,'difference':.12},'decision':decision}
 (ART/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');(ART/'posthoc_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
 c={x['comparison']:x for x in comparisons};report=['# Early-trajectory accumulation adjudication','',f"Integrity PASS: {integrity['pass']} (`50/50` pairs, `300/300` episodes, invalid 0).",'','## Success rates']+[f'- {a}: {rates[a]:.1%}' for a in ARMS]+['','## Paired effects']
 for x in comparisons:report.append(f"- {x['comparison']}: {x['difference']:+.1%}, 95% CI [{x['bootstrap_95_ci'][0]:+.1%}, {x['bootstrap_95_ci'][1]:+.1%}], discordant {x['first_only_success']}/{x['second_only_success']}, McNemar p={x['exact_mcnemar_p']:.4g}.")
 report+=['','## Decision',decision['status'],decision['reason']];(ART/'report.md').write_text('\n'.join(report)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
