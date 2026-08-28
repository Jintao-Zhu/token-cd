#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json,math,random
from collections import defaultdict
from pathlib import Path
ARMS=('vanilla','random8_common','attention8_common','instability8_common')
PRIMARY=(('instability8_common','random8_common'),('attention8_common','random8_common'),('instability8_common','attention8_common'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def comparison(rows,a,b,seed):
 delta=[r[a]-r[b] for r in rows];rng=random.Random(seed);boot=sorted(sum(delta[rng.randrange(len(delta))] for _ in delta)/len(delta) for _ in range(10000));a_only=sum(r[a] and not r[b] for r in rows);b_only=sum(r[b] and not r[a] for r in rows);n=a_only+b_only;k=min(a_only,b_only);p=min(1.,2*sum(math.comb(n,i) for i in range(k+1))*.5**n) if n else 1.
 return {'comparison':f'{a}_minus_{b}','difference':sum(delta)/len(delta),'paired_bootstrap_95_ci':[boot[249],boot[9749]],'first_only_success':a_only,'second_only_success':b_only,'exact_mcnemar_p':p}
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();art=a.artifact.resolve();protocol_hash=sha(art/'protocol.lock.yaml');by=defaultdict(dict);bad=[];max_match_error=0.;guided_steps=0
 files=sorted((art/'episodes').glob('*.json'))
 for path in files:
  x=json.loads(path.read_text());key=(x['pair_id'],x['arm'])
  if x['arm'] in by[x['pair_id']]:bad.append(f'duplicate:{key}')
  by[x['pair_id']][x['arm']]=x
  if x['status']!='complete' or not x['all_actions_finite'] or x['protocol_sha256']!=protocol_hash:bad.append(x['episode_id'])
  for trace in x['replan_traces']:
   if len(trace['selected_indices'])!=8 or sorted(trace['selected_indices'])!=sorted(trace['changed_indices']) or not trace['protected_tokens_untouched'] or not trace['all_output_finite']:bad.append(f"trace:{x['episode_id']}:{trace['replan']}")
   for step in trace['step_traces']:
    guided_steps+=1;error=abs(step['applied_guidance_norm']-step['common_norm'])/max(step['common_norm'],1e-12);max_match_error=max(max_match_error,error)
    if step['alpha']>1+1e-8 or step['extrapolated'] or error>1e-5:bad.append(f"common:{x['episode_id']}:{trace['replan']}:{step['step']}")
 rows=[];raw_camera_mismatch=0
 for pair,arms in sorted(by.items()):
  if set(arms)!=set(ARMS):bad.append(f'arms:{pair}');continue
  if len({json.dumps(arms[x]['initial_gate'],sort_keys=True) for x in ARMS})!=1:bad.append(f'initial:{pair}');continue
  raw_camera_mismatch+=int(any(arms['vanilla']['raw_reset_mismatch_flags'].values()));rows.append({'pair_id':pair,'task_id':4,'init_state_id':arms['vanilla']['init_state_id'],**{arm:int(arms[arm]['success']) for arm in ARMS}})
 integrity={'pass':len(files)==200 and len(rows)==50 and not bad and not list((art/'invalid_pairs').glob('*.json')),'episodes':len(files),'paired_states':len(rows),'each_arm':{arm:sum(arm in x for x in by.values()) for arm in ARMS},'invalid_pairs':len(list((art/'invalid_pairs').glob('*.json'))),'raw_reset_camera_mismatch_pairs':raw_camera_mismatch,'guided_flow_steps_audited':guided_steps,'max_applied_common_relative_error':max_match_error,'bad':bad}
 if not integrity['pass']:
  (art/'summary.json').write_text(json.dumps({'integrity':integrity,'decision':'FAILED_INTEGRITY'},indent=2)+'\n');raise RuntimeError('integrity failed')
 with (art/'paired_results.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 rates={arm:sum(r[arm] for r in rows)/len(rows) for arm in ARMS};comparisons=[comparison(rows,x,y,2026081000+i) for i,(x,y) in enumerate(PRIMARY)];vs_vanilla=[comparison(rows,arm,'vanilla',2026081100+i) for i,arm in enumerate(ARMS[1:])]
 ai=next(x for x in comparisons if x['comparison']=='attention8_common_minus_random8_common');ii=next(x for x in comparisons if x['comparison']=='instability8_common_minus_random8_common')
 if ai['paired_bootstrap_95_ci'][0]>0 or ii['paired_bootstrap_95_ci'][0]>0:decision='TOKEN_IDENTITY_UTILITY_SUPPORTED_UNDER_COMMON_MAGNITUDE'
 elif abs(ai['difference'])<=.04 and abs(ii['difference'])<=.04:decision='COMMON_MAGNITUDE_SELECTORS_SIMILAR'
 else:decision='COMMON_MAGNITUDE_SELECTOR_RESULT_UNCERTAIN'
 summary={'integrity':integrity,'success_rates':rates,'primary_comparisons':comparisons,'secondary_vs_vanilla':vs_vanilla,'decision':decision};(art/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');(art/'decision.json').write_text(json.dumps({'decision':decision,'integrity':'PASS'},indent=2)+'\n')
 report=['# Perturbation-Norm-Matched Selector Test','',f"Integrity: PASS (`{len(rows)}/50` pairs, `{len(files)}/200` episodes, 0 invalid).",f"Audited guided flow steps: `{guided_steps}`; maximum applied/common relative error: `{max_match_error:.3g}`.",'','## Success rates']+[f'- {arm}: {rates[arm]:.1%}' for arm in ARMS]+['','## Primary paired comparisons']+[f"- {x['comparison']}: {x['difference']:+.1%}, 95% CI [{x['paired_bootstrap_95_ci'][0]:+.1%}, {x['paired_bootstrap_95_ci'][1]:+.1%}], discordant {x['first_only_success']} vs {x['second_only_success']}, exact McNemar p={x['exact_mcnemar_p']:.4g}" for x in comparisons]+['','## Versus Vanilla']+[f"- {x['comparison']}: {x['difference']:+.1%}, 95% CI [{x['paired_bootstrap_95_ci'][0]:+.1%}, {x['paired_bootstrap_95_ci'][1]:+.1%}], exact McNemar p={x['exact_mcnemar_p']:.4g}" for x in vs_vanilla]+['','## Decision',decision,'','This is a Task-4 development experiment over all 50 official init states, not an independent confirmation.'];(art/'report.md').write_text('\n'.join(report)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
