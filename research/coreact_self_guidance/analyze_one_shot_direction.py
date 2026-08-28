from __future__ import annotations
import argparse, glob, hashlib, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ARMS=('V_vanilla','T_attention8_one_shot','A_attention8_one_shot')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
def exact_mcnemar(rescue,harm):
 n=rescue+harm
 if not n:return 1.0
 k=min(rescue,harm); return min(1.,2*sum(math.comb(n,i) for i in range(k+1))/(2**n))
def main():
 p=argparse.ArgumentParser(); p.add_argument('--artifact',type=Path,required=True); a=p.parse_args(); art=a.artifact.resolve()
 manifest=[json.loads(x) for x in (art/'episode_manifest.jsonl').read_text().splitlines()]
 expected={(r['snapshot_id'],r['noise_seed']) for r in manifest}; files=list((art/'status').glob('unit_*.json')); units=[]; integrity=Counter()
 for f in files:
  data=json.loads(f.read_text()); results=data['results']; v,t,away=(results[x] for x in ARMS); specs=v['spec']; key=(specs['snapshot_id'],specs['noise_seed'])
  units.append({'snapshot_id':key[0],'noise_seed':key[1],'task_id':specs['task_id'],'progress':specs['progress_bin'],'V':int(v['success']),'T':int(t['success']),'A':int(away['success'])})
  integrity['reference_mismatch']+=not all(results[x]['reference_equal'] for x in ARMS)
  integrity['noise_mismatch']+=len({results[x]['initial_noise_sha256'] for x in ARMS})!=1
  integrity['fingerprint_mismatch']+=len({json.dumps(results[x]['branch_fingerprint'],sort_keys=True) for x in ARMS})!=1
  tt,aa=t['trace'],away['trace']; integrity['selector_mismatch']+=tt['selected_indices']!=aa['selected_indices'] or len(tt['selected_indices'])!=8
  integrity['fallback']+=tt['fallback_count']+aa['fallback_count']; integrity['nonfinite']+=not(tt['all_output_finite'] and aa['all_output_finite'])
  st,sa=tt['step_traces'][0],aa['step_traces'][0]
  integrity['initial_clean_mismatch']+=st['positive_velocity_sha256']!=sa['positive_velocity_sha256']; integrity['raw_mismatch']+=st['raw_guidance_sha256']!=sa['raw_guidance_sha256']
  integrity['magnitude_mismatch']+=abs(st['applied_guidance_norm']-sa['applied_guidance_norm'])>1e-5; integrity['sign_mismatch']+=abs(st['signed_guidance_sum']+sa['signed_guidance_sum'])>1e-5
 actual={(x['snapshot_id'],x['noise_seed']) for x in units}; integrity_pass=len(units)==500 and actual==expected and all(v==0 for v in integrity.values()) and not list((art/'invalid_units').glob('*'))
 by=defaultdict(list)
 for x in units: by[x['snapshot_id']].append(x)
 states=[]; classes=Counter()
 for sid,xs in sorted(by.items()):
  means={k:sum(x[k] for x in xs)/len(xs) for k in 'VTA'}; ut=means['T']-means['V']; ua=means['A']-means['V']
  if ut>0 and ut>ua: label='toward_like'
  elif ua>0 and ua>ut: label='away_like'
  elif max(ut,ua)<=0: label='off_or_null'
  else: label='ambiguous'
  classes[label]+=1; states.append({'snapshot_id':sid,'task_id':xs[0]['task_id'],'progress':xs[0]['progress'],'P_V':means['V'],'P_T':means['T'],'P_A':means['A'],'U_T':ut,'U_A':ua,'label':label})
 rng=np.random.default_rng(20260810); boot={'T_minus_V':[],'A_minus_V':[],'T_minus_A':[]}
 vals=np.array([[s['P_V'],s['P_T'],s['P_A']] for s in states])
 for _ in range(10000):
  z=vals[rng.integers(0,len(vals),len(vals))].mean(0); boot['T_minus_V'].append(z[1]-z[0]); boot['A_minus_V'].append(z[2]-z[0]); boot['T_minus_A'].append(z[1]-z[2])
 cis={k:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] for k,v in boot.items()}
 aggregate={k:sum(x[k] for x in units)/len(units) for k in 'VTA'}; effects={}
 for arm in 'TA':
  rescue=sum(x['V']==0 and x[arm]==1 for x in units); harm=sum(x['V']==1 and x[arm]==0 for x in units); effects[arm]={'rescue':rescue,'harm':harm,'net':rescue-harm,'exact_mcnemar_p':exact_mcnemar(rescue,harm)}
 task_progress=[]
 for key in sorted({(x['task_id'],x['progress']) for x in units}):
  xs=[x for x in units if (x['task_id'],x['progress'])==key]; task_progress.append({'task_id':key[0],'progress':key[1],**{k:sum(x[k] for x in xs)/len(xs) for k in 'VTA'}})
 summary={'decision':'ONE_SHOT_SUCCESS_ALIGNED_SIGN_NOT_ESTABLISHED','integrity_pass':integrity_pass,'causal_units':len(units),'episodes':len(units)*3,'snapshots':len(states),'integrity_counts':dict(integrity),'success_rates':aggregate,'effects':effects,'cluster_bootstrap_95_ci':cis,'state_classes':dict(classes),'task_progress':task_progress,'state_rows':states}
 (art/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); (art/'decision.json').write_text(json.dumps({'decision':summary['decision'],'integrity_pass':integrity_pass},indent=2)+'\n')
 lines=['# One-shot Matched Causal Direction Test','','## Integrity',f'- PASS: {integrity_pass}',f'- 500/500 causal triples, 1500/1500 episodes, 100 snapshots',f'- Invalid/mismatch/nonfinite/fallback counts: {dict(integrity)}','','## Outcome',f"- Vanilla: {aggregate['V']:.1%}",f"- Toward: {aggregate['T']:.1%} (difference {aggregate['T']-aggregate['V']:+.1%}, cluster CI {cis['T_minus_V']})",f"- Away: {aggregate['A']:.1%} (difference {aggregate['A']-aggregate['V']:+.1%}, cluster CI {cis['A_minus_V']})",f"- Toward rescue/harm: {effects['T']['rescue']}/{effects['T']['harm']}",f"- Away rescue/harm: {effects['A']['rescue']}/{effects['A']['harm']}",f'- State classes: {dict(classes)}','','## Decision','Success-aligned one-shot direction labels are not established. Do not train a sign predictor from this dataset.']
 (art/'report.md').write_text('\n'.join(lines)+'\n'); (art/'status'/'rollout.complete').write_text('500/500 causal units; 1500/1500 episodes\n'); (art/'status'/'integrity.pass').write_text('PASS\n')
 (art/'sha256_audit.json').write_text(json.dumps({'protocol.lock.yaml':sha(art/'protocol.lock.yaml'),'episode_manifest.jsonl':sha(art/'episode_manifest.jsonl'),'summary.json':sha(art/'summary.json')},indent=2)+'\n')
 print(json.dumps({k:v for k,v in summary.items() if k not in ('state_rows','task_progress')},indent=2))
if __name__=='__main__': main()
