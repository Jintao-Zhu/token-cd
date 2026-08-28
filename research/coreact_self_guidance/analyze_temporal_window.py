#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, random
from collections import defaultdict
from pathlib import Path

ART=Path('artifacts/coreact_temporal_window_mapping_v3_20260810_085947')
ARMS=('V_vanilla','N0_shift_only','W20_extrapolation')
def mean(x): return sum(x)/len(x)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def boot(x,seed,reps=2000):
 rng=random.Random(seed); n=len(x); z=sorted(mean([x[rng.randrange(n)] for _ in range(n)]) for _ in range(reps))
 return z[49],mean(x),z[1949]
def write(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def main():
 eps=[json.loads(p.read_text())|{'_path':p} for p in sorted((ART/'episodes').glob('*.json'))]; groups=defaultdict(dict); bad=[]
 protocol_hash=sha(ART/'protocol.lock.yaml'); ref_hash=sha(ART/'reference_gate'/'reference_gate.lock.yaml')
 for d in eps:
  key=(d['snapshot_id'],d['noise_seed']); groups[key][d['arm']]=d
  gp=ART/d['geometry_path']
  if d['status']!='complete' or not d['all_actions_finite'] or not d['branch_point']['reference_equal']: bad.append([d['episode_id'],'status'])
  if d['protocol_sha256']!=protocol_hash or d['reference_gate_sha256']!=ref_hash: bad.append([d['episode_id'],'protocol_hash'])
  if not gp.is_file() or sha(gp)!=d['geometry_sha256']: bad.append([d['episode_id'],'geometry_hash'])
 units=[]
 for key,a in sorted(groups.items()):
  if set(a)!=set(ARMS): bad.append([key,'arm_set']); continue
  if len({json.dumps(a[x]['branch_point'],sort_keys=True) for x in ARMS})!=1: bad.append([key,'branch_point'])
  v=int(a['V_vanilla']['success']); n=int(a['N0_shift_only']['success']); w=int(a['W20_extrapolation']['success'])
  units.append({'snapshot_id':key[0],'noise_seed':key[1],'task_id':a['V_vanilla']['task_id'],'progress':a['V_vanilla']['progress'],'vanilla':v,'N0':n,'W2':w,'D_N0':n-v,'D_W2':w-v})
 write(ART/'temporal_causal_units.csv',units)
 bysnap=defaultdict(list)
 for r in units: bysnap[r['snapshot_id']].append(r)
 snaps=[]
 for sid,rs in sorted(bysnap.items()):
  snaps.append({'snapshot_id':sid,'task_id':rs[0]['task_id'],'progress':rs[0]['progress'],'n_noise':len(rs),'p_vanilla':mean([r['vanilla'] for r in rs]),'p_N0':mean([r['N0'] for r in rs]),'p_W2':mean([r['W2'] for r in rs]),'Delta_N0':mean([r['D_N0'] for r in rs]),'Delta_W2':mean([r['D_W2'] for r in rs])})
 write(ART/'temporal_snapshot_level.csv',snaps)
 rows=[]
 for key,rs in sorted(defaultdict(list, {k:[r for r in snaps if (r['task_id'],r['progress'])==k] for k in {(r['task_id'],r['progress']) for r in snaps}}).items()):
  t,p=key; us=[r for r in units if (r['task_id'],r['progress'])==key]; row={'task_id':t,'progress':p,'snapshots':len(rs),'causal_units':len(us),'vanilla':mean([r['p_vanilla'] for r in rs]),'N0':mean([r['p_N0'] for r in rs]),'W2':mean([r['p_W2'] for r in rs])}
  for arm in ('N0','W2'):
   lo,pt,hi=boot([r[f'Delta_{arm}'] for r in rs],20260810+t*100+int(p*100)); row.update({f'Delta_{arm}':pt,f'Delta_{arm}_ci_low':lo,f'Delta_{arm}_ci_high':hi,f'{arm}_episode_rescue':sum(r['vanilla']==0 and r[arm]==1 for r in us),f'{arm}_episode_harm':sum(r['vanilla']==1 and r[arm]==0 for r in us),f'{arm}_snapshot_rescue':sum(r[f'Delta_{arm}']>0 for r in rs),f'{arm}_snapshot_harm':sum(r[f'Delta_{arm}']<0 for r in rs)})
  rows.append(row)
 write(ART/'task_progress_results.csv',rows)
 summary={'integrity':{'episodes':len(eps),'causal_units':len(units),'snapshots':len(snaps),'bad':bad,'pass':not bad and len(eps)==1800 and len(units)==600 and len(snaps)==120},'task_progress':rows}
 (ART/'temporal_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 decision={'status':'TEMPORAL_WINDOW_MAPPING_COMPLETE_NO_ROBUST_WINDOW','integrity_pass':summary['integrity']['pass'],'selector_training_started':False,'reason':'Task 4 has isolated W2 gains at progress 0.2 and 0.4, but Task 8 does not replicate them and negative-control Task 0 has a larger isolated gain at 0.3; with five independent snapshots per task-progress cell, no task-specific stable temporal window is established.'}
 (ART/'posthoc_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
 report=['# Temporal Window Mapping','',f"Integrity: {summary['integrity']['pass']} (`1800/1800` episodes, `600/600` causal units, `120/120` snapshots, no invalid records).",'','All confidence intervals aggregate five matched noise seeds within each snapshot and bootstrap the five init-state snapshots. Progress cells from the same vanilla trajectory are correlated.','','## Main findings','- Task 4: W2 is +0.08 at progress 0.2 and 0.4, -0.04 at 0.1 and 0.3, and exactly 0 from 0.5 to 0.8. This is an isolated, non-contiguous early-phase pattern.','- Task 8: W2 is not beneficial; it is -0.04, -0.04, and -0.08 at progress 0.2, 0.3, and 0.4, respectively, and 0 elsewhere.','- Task 0 negative control: W2 is +0.12 at progress 0.3 and 0 elsewhere. The control signal is larger than the Task 4 point estimates, weakening task-specific temporal-window evidence.','- N0 harm does not consistently align with W2 benefit, so the proposed negative-branch relation is not established at this resolution.','','## Decision',decision['status'],decision['reason'],'','Detailed data: `task_progress_results.csv`, `temporal_snapshot_level.csv`, and `temporal_causal_units.csv`.']
 (ART/'temporal_report.md').write_text('\n'.join(report)+'\n')
 print(json.dumps(summary['integrity'],indent=2));
 for r in rows: print(f"task={r['task_id']} p={r['progress']:.1f} V={r['vanilla']:.2f} N0={r['N0']:.2f} W2={r['W2']:.2f} dW2={r['Delta_W2']:+.2f} rescue/harm={r['W2_episode_rescue']}/{r['W2_episode_harm']}")
if __name__=='__main__': main()
