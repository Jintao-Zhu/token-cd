#!/usr/bin/env python3
"""Integrity and paired-outcome audit for the frozen Raw-L11/Spatial-Null study."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np

ROOT=Path('/home/leju-suzhou/zjt_ws/token-cd')

def exact_mcnemar(w,l):
    n=w+l
    if n==0:return 1.0
    k=min(w,l)
    return min(1.0,2.0*sum(math.comb(n,i) for i in range(k+1))/(2**n))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--artifact',type=Path,required=True); ap.add_argument('--preflight',action='store_true'); a=ap.parse_args()
    art=a.artifact.resolve(); manifest=json.loads((art/'cases_manifest.json').read_text()); prior=json.loads((art/'position_prior_frozen.json').read_text())
    expected=1 if a.preflight else len(manifest['episodes'])
    cases=manifest['episodes'][:1] if a.preflight else manifest['episodes']
    outcomes=[]; checked=[]; errors=[]
    for c in cases:
      eps={arm:art/'episodes'/c['case_id']/arm/'episode.json' for arm in ('vanilla','matched','matched_spatial_null')}
      if not all(p.is_file() for p in eps.values()):
        errors.append({'case_id':c['case_id'],'error':'missing arm episode','paths':{k:v.exists() for k,v in eps.items()}}); continue
      d={k:json.loads(v.read_text()) for k,v in eps.items()}
      arrays={k:np.load(v.parent/'step_arrays.npz') for k,v in eps.items()}
      states={x['initial_state_sha256'] for x in d.values()}
      first_rgb={x['trace'][0]['rgb_sha256'] for x in d.values()}
      first_state={x['trace'][0]['sim_state_sha256'] for x in d.values()}
      ms={int(x['trace'][0]['matched_m']) for x in d.values()}
      clean=[arrays[k]['positive_action_logits'][0] for k in d]
      attention=[arrays[k]['l11_attention_scores'][0] for k in d]
      ok=(len(states)==len(first_rgb)==len(first_state)==len(ms)==1 and
          all(np.array_equal(clean[0],x) for x in clean[1:]) and all(np.array_equal(attention[0],x) for x in attention[1:]))
      if not ok: errors.append({'case_id':c['case_id'],'error':'initial state/RGB/m/logit/attention mismatch'})
      raw=d['matched']['trace'][0]; nul=d['matched_spatial_null']['trace'][0]; m=int(raw['matched_m'])
      canonical=set(int(x) for x in np.argsort(-attention[1],kind='stable')[:m])
      raw_ids=set(int(x) for x in raw['selected_token_ids'])
      null_ids=set(int(x) for x in nul['selected_token_ids'])
      if raw_ids!=canonical: errors.append({'case_id':c['case_id'],'error':'Raw L11 selection differs from stored attention Top-m'})
      if len(null_ids)!=m or int(nul['m_used'])!=m: errors.append({'case_id':c['case_id'],'error':'Spatial Null did not select exactly m_t'})
      payload=f"{int(c['task_id'])}:{int(c['case_seed'])}:0:N1".encode(); seed=int.from_bytes(hashlib.sha256(payload).digest()[:8],'big')
      p=np.asarray(prior['position_priors'][str(c['task_id'])],dtype=np.float64).reshape(-1); p/=p.sum()
      expected_null=set(np.random.default_rng(seed).choice(256,size=m,replace=False,p=p).tolist())
      if null_ids!=expected_null: errors.append({'case_id':c['case_id'],'error':'Spatial Null first-step mask is not reproducible from frozen prior/seed'})
      step_mask_errors=[]
      for arm_name in ('matched','matched_spatial_null'):
        ep=d[arm_name]
        for step_i,tr in enumerate(ep['trace']):
          step_m=int(tr['matched_m']); selected_step=set(int(x) for x in tr['selected_token_ids'])
          if len(selected_step)!=step_m or int(tr['m_used'])!=step_m:
            step_mask_errors.append({'arm':arm_name,'step':step_i,'kind':'count'})
            continue
          if arm_name=='matched':
            att=arrays[arm_name]['l11_attention_scores'][step_i]
            expected_step=set(int(x) for x in np.argsort(-att,kind='stable')[:step_m])
          else:
            payload_i=f"{int(c['task_id'])}:{int(c['case_seed'])}:{step_i}:N1".encode()
            seed_i=int.from_bytes(hashlib.sha256(payload_i).digest()[:8],'big')
            prior_i=np.asarray(prior['position_priors'][str(c['task_id'])],dtype=np.float64).reshape(-1); prior_i/=prior_i.sum()
            expected_step=set(np.random.default_rng(seed_i).choice(256,size=step_m,replace=False,p=prior_i).tolist())
            if tr.get('spatial_null_seed')!=seed_i:
              step_mask_errors.append({'arm':arm_name,'step':step_i,'kind':'seed'})
          if selected_step!=expected_step:
            step_mask_errors.append({'arm':arm_name,'step':step_i,'kind':'selection'})
      if step_mask_errors: errors.append({'case_id':c['case_id'],'error':'per-step selector audit failure','details':step_mask_errors[:5]})
      checked.append({'case_id':c['case_id'],'task_id':int(c['task_id']),'initial_hash_match':len(states)==1,
        'first_rgb_match':len(first_rgb)==1,'first_clean_logits_exact':all(np.array_equal(clean[0],x) for x in clean[1:]),
        'first_attention_exact':all(np.array_equal(attention[0],x) for x in attention[1:]),'m':m,
        'raw_mask_exact':raw_ids==canonical,'null_mask_exact_m':len(null_ids)==m,'null_mask_seed_exact':null_ids==expected_null,
        'raw_and_null_all_steps_exact':not bool(step_mask_errors),'raw_and_null_steps':min(len(d['matched']['trace']),len(d['matched_spatial_null']['trace'])),
        'success':{k:bool(v['success']) for k,v in d.items()}})
      outcomes.append({'task_id':int(c['task_id']),'vanilla':bool(d['vanilla']['success']),'raw':bool(d['matched']['success']),'null':bool(d['matched_spatial_null']['success'])})
    if not a.preflight and len(checked)!=expected: errors.append({'error':f'completed pair count {len(checked)} != {expected}'})
    def compare(x,y):
      w=sum(r[x] and not r[y] for r in outcomes); l=sum(r[y] and not r[x] for r in outcomes)
      return {'x_successes':sum(r[x] for r in outcomes),'y_successes':sum(r[y] for r in outcomes),
        'x_only':w,'y_only':l,'ties':len(outcomes)-w-l,'net_x_minus_y':w-l,'mcnemar_exact_two_sided_p':exact_mcnemar(w,l)}
    per_task={}
    tids=sorted({r['task_id'] for r in outcomes})
    for tid in tids:
      sub=[r for r in outcomes if r['task_id']==tid]
      per_task[str(tid)]={'n':len(sub),'vanilla_success':sum(r['vanilla'] for r in sub),'raw_success':sum(r['raw'] for r in sub),'null_success':sum(r['null'] for r in sub),
        'raw_only_vs_null':sum(r['raw'] and not r['null'] for r in sub),'null_only_vs_raw':sum(r['null'] and not r['raw'] for r in sub),
        'raw_minus_null_rate':float(np.mean([r['raw']-r['null'] for r in sub]))}
    macro=float(np.mean([v['raw_minus_null_rate'] for v in per_task.values()])) if per_task else None
    result={'protocol_id':manifest['protocol_id'],'audit_mode':'preflight' if a.preflight else 'final',
      'status':'PASS' if not errors and len(checked)==expected else 'FAIL','expected_pairs':expected,'checked_pairs':len(checked),
      'arm_episode_records':len(checked)*3,'errors':errors,'primary_raw_vs_spatial_null':compare('raw','null'),
      'vanilla_context_raw_vs_vanilla':compare('raw','vanilla'),'vanilla_context_null_vs_vanilla':compare('null','vanilla'),
      'task_results':per_task,'five_task_macro_raw_minus_null_rate':macro,'pairs':checked}
    out=art/('PREFLIGHT_AUDIT.json' if a.preflight else 'FINAL_AUDIT.json'); out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','errors')},indent=2)); print('errors='+json.dumps(errors[:10]))
    if errors or len(checked)!=expected: raise SystemExit(2)

if __name__=='__main__': main()
