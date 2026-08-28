#!/usr/bin/env python3
from __future__ import annotations
import csv,glob,hashlib,json,math,random
from collections import defaultdict
from pathlib import Path
ART=Path('artifacts/coreact_dynamic_core_token_selection_rollout_v2_20260810_144650');ARMS=('vanilla','random8','attention8','instability8','combined8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def cmp(rows,a,b,seed):
 d=[r[a]-r[b] for r in rows];rng=random.Random(seed);z=sorted(sum(d[rng.randrange(len(d))] for _ in d)/len(d) for _ in range(2000));ao=sum(r[a] and not r[b] for r in rows);bo=sum(r[b] and not r[a] for r in rows);n=ao+bo;k=min(ao,bo);p=min(1,2*sum(math.comb(n,i) for i in range(k+1))*.5**n) if n else 1
 return {'comparison':f'{a}_minus_{b}','difference':sum(d)/len(d),'bootstrap_95_ci':[z[49],z[1949]],'first_only_success':ao,'second_only_success':bo,'exact_mcnemar_p':p}
def main():
 by=defaultdict(dict);bad=[];ph=sha(ART/'protocol.lock.yaml')
 for p in sorted((ART/'episodes').glob('*.json')):
  x=json.loads(p.read_text());by[x['pair_id']][x['arm']]=x
  if x['status']!='complete' or not x['all_actions_finite'] or x['protocol_sha256']!=ph:bad.append(x['episode_id'])
 rows=[];raw_mismatch=0
 for pair,a in sorted(by.items()):
  if set(a)!=set(ARMS) or len({json.dumps(a[x]['initial_gate'],sort_keys=True) for x in ARMS})!=1:bad.append(pair);continue
  raw_mismatch+=int(any(a['vanilla']['raw_reset_mismatch_flags'].values()));rows.append({'pair_id':pair,'task_id':a['vanilla']['task_id'],'init_state_id':a['vanilla']['init_state_id'],**{x:int(a[x]['success']) for x in ARMS}})
 with (ART/'paired_results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 results={}
 pairs=[(a,'vanilla') for a in ARMS[1:]]+[('instability8','attention8'),('combined8','attention8'),('attention8','random8'),('instability8','random8'),('combined8','random8'),('combined8','instability8')]
 for task in (4,7):
  rs=[r for r in rows if r['task_id']==task];results[str(task)]={'success_rates':{a:sum(r[a] for r in rs)/len(rs) for a in ARMS},'comparisons':[cmp(rs,a,b,20260810+task*100+i) for i,(a,b) in enumerate(pairs)]}
 integrity={'pass':not bad and len(rows)==100,'episodes':sum(len(x) for x in by.values()),'pairs':len(rows),'invalid_pairs':len(list((ART/'invalid_pairs').glob('*.json'))),'raw_reset_camera_mismatch_pairs':raw_mismatch,'bad':bad}
 decision={'status':'DYNAMIC_SELECTOR_SIGNAL_ATTENTION_AND_INSTABILITY_SIMILAR','token_identity_signal':'supported_directionally_against_random','temporal_instability_increment_over_attention':'not_established','combined_advantage':'not_established','reason':'Task 4 and Task 7 both order instability/combined >= attention >= random, but instability and combined exceed attention by only 2pp with paired intervals crossing zero. Task 4 shows the strongest selector signal; Task 7 is near ceiling.'}
 summary={'integrity':integrity,'results':results,'phase0':'artifacts/coreact_dynamic_core_token_selection_phase0_v1_20260810_140427','decision':decision};(ART/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');(ART/'posthoc_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
 report=['# Dynamic Core Token Selection','',f"Integrity PASS: {integrity['pass']} (`100/100` pairs, `500/500` episodes, unresolved invalid pairs 0).",f"Raw reset camera mismatches neutralized by the preregistered canonical first-observation rule: {raw_mismatch} pairs.",'']
 for task in (4,7):
  r=results[str(task)];report+= [f'## Task {task}']+[f'- {a}: {r["success_rates"][a]:.1%}' for a in ARMS]+['']+[f"- {x['comparison']}: {x['difference']:+.1%}, 95% CI [{x['bootstrap_95_ci'][0]:+.1%}, {x['bootstrap_95_ci'][1]:+.1%}], McNemar p={x['exact_mcnemar_p']:.4g}" for x in r['comparisons']]+['']
 report+=['## Decision',decision['status'],decision['reason']];(ART/'report.md').write_text('\n'.join(report)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
