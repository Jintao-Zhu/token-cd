from __future__ import annotations
import argparse,csv,hashlib,json,math,random
from collections import defaultdict
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
ARMS=('vanilla','random8_common','attention8_common')
def compare(rows,a,b,seed):
 d=[r[a]-r[b] for r in rows]; rng=random.Random(seed); boots=sorted(sum(d[rng.randrange(len(d))] for _ in d)/len(d) for _ in range(10000)); ao=sum(r[a] and not r[b] for r in rows); bo=sum(r[b] and not r[a] for r in rows); n=ao+bo; k=min(ao,bo); p=min(1.,2*sum(math.comb(n,i) for i in range(k+1))*.5**n) if n else 1.
 return {'comparison':f'{a}_minus_{b}','difference':sum(d)/len(d),'cluster_bootstrap_95_ci':[boots[249],boots[9749]],'first_only_success':ao,'second_only_success':bo,'exact_mcnemar_p':p}
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);a=p.parse_args(); art=a.artifact.resolve(); protocol_hash=sha(art/'protocol.lock.yaml'); files=sorted((art/'episodes').glob('*.json')); by=defaultdict(dict); bad=[]
 for f in files:
  x=json.loads(f.read_text()); by[x['pair_id']][x['arm']]=x
  if x['status']!='complete' or not x['all_actions_finite'] or x['protocol_sha256']!=protocol_hash:bad.append(x['episode_id'])
  for tr in x.get('replan_traces',[]):
   if len(tr['selected_indices'])!=8 or sorted(tr['selected_indices'])!=sorted(tr['changed_indices']) or not tr['protected_tokens_untouched'] or not tr['all_output_finite']:bad.append('trace:'+x['episode_id'])
   for step in tr['step_traces']:
    err=abs(step['applied_guidance_norm']-step['common_norm'])/max(step['common_norm'],1e-12)
    if step['alpha']>1+1e-8 or step['extrapolated'] or err>1e-5:bad.append('magnitude:'+x['episode_id'])
 rows=[]
 for pair,arms in sorted(by.items()):
  if set(arms)!=set(ARMS):bad.append('arms:'+pair);continue
  if len({json.dumps(arms[x]['initial_gate'],sort_keys=True) for x in ARMS})!=1:bad.append('initial:'+pair);continue
  rows.append({'pair_id':pair,'task_id':arms['vanilla']['task_id'],'init_state_id':arms['vanilla']['init_state_id'],**{arm:int(arms[arm]['success']) for arm in ARMS}})
 integrity={'pass':len(files)==1350 and len(rows)==450 and not bad and not list((art/'invalid_pairs').glob('*')),'episodes':len(files),'paired_states':len(rows),'each_arm':{arm:sum(arm in x for x in by.values()) for arm in ARMS},'invalid_pairs':len(list((art/'invalid_pairs').glob('*'))),'bad_count':len(bad)}
 if not integrity['pass']:
  (art/'summary.json').write_text(json.dumps({'integrity':integrity,'decision':'HELDOUT_CONFIRMATION_INTEGRITY_FAILED'},indent=2)+'\n');raise RuntimeError('integrity failed')
 with (art/'paired_results.csv').open('w',newline='') as f: csv.DictWriter(f,fieldnames=list(rows[0])).writerows(rows)
 rates={arm:sum(r[arm] for r in rows)/len(rows) for arm in ARMS}; comps=[compare(rows,'attention8_common','random8_common',20260810),compare(rows,'attention8_common','vanilla',20260811),compare(rows,'random8_common','vanilla',20260812)]
 task=[]
 for t in sorted({r['task_id'] for r in rows}):
  xs=[r for r in rows if r['task_id']==t]; task.append({'task_id':t,'n':len(xs),**{arm:sum(r[arm] for r in xs)/len(xs) for arm in ARMS},'attention_minus_random':sum(r['attention8_common']-r['random8_common'] for r in xs)/len(xs)})
 decision='HELDOUT_ATTENTION_IDENTITY_SUPPORTED' if comps[0]['cluster_bootstrap_95_ci'][0]>0 and sum(x['attention_minus_random']>0 for x in task)>=5 else 'HELDOUT_ATTENTION_IDENTITY_NOT_CONFIRMED'
 summary={'integrity':integrity,'success_rates':rates,'primary_comparison':comps[0],'secondary_comparisons':comps[1:],'task_results':task,'decision':decision}; (art/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); (art/'decision.json').write_text(json.dumps({'decision':decision,'integrity':'PASS'},indent=2)+'\n'); (art/'status'/'rollout.complete').write_text('450/450 pairs; 1350/1350 episodes\n')
 lines=['# Held-out Common-Min Attention Confirmation','',f"Integrity: PASS (`{len(rows)}/450` pairs, `{len(files)}/1350` episodes).",'','## Success rates']+[f'- {a}: {rates[a]:.1%}' for a in ARMS]+['','## Attention vs Random',f"- Difference: {comps[0]['difference']:+.1%}",f"- Cluster bootstrap 95% CI: {comps[0]['cluster_bootstrap_95_ci']}",f"- Discordant pairs: Attention-only {comps[0]['first_only_success']}, Random-only {comps[0]['second_only_success']}",f"- Exact McNemar p: {comps[0]['exact_mcnemar_p']:.4g}",'','## Task direction']+[f"- Task {x['task_id']}: V={x['vanilla']:.1%}, R={x['random8_common']:.1%}, A={x['attention8_common']:.1%}, A-R={x['attention_minus_random']:+.1%}" for x in task]+['','## Decision',decision]
 (art/'report.md').write_text('\n'.join(lines)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
