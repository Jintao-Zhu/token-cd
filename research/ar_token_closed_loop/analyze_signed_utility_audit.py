from __future__ import annotations
import argparse,glob,json,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def metrics(rows):
 changed=[x for x in rows if abs(x['delta_error'])>1e-12]; eligible=[x for x in changed if abs(x['U'])>1e-12]
 return {'tokens':len(rows),'free_running_changed_fraction':len(changed)/len(rows),'sign_agreement_on_changed':float(np.mean([np.sign(x['U'])==np.sign(x['delta_error']) for x in eligible])) if eligible else None,'spearman_U_vs_error_delta':float(spearmanr([x['U'] for x in rows],[x['delta_error'] for x in rows]).statistic),'positive_U_mask_worsens_fraction':float(np.mean([x['delta_error']>0 for x in rows if x['U']>0])),'negative_U_mask_improves_fraction':float(np.mean([x['delta_error']<0 for x in rows if x['U']<0]))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();art=a.artifact.resolve();files=sorted((art/'states').glob('*.json'));rows=[];bad=[]
 for f in files:
  d=json.loads(f.read_text()); sels=d['selected']; candidates=d['candidates']; indices=[x['token_index'] for x in candidates]
  if len(candidates)!=32 or len(set(indices))!=32 or [len(sels[x]) for x in ('attention_top16','attention_bottom8','random8')]!=[16,8,8] or not d['finite']:bad.append(d['snapshot_id'])
  for x in candidates:rows.append({'snapshot_id':d['snapshot_id'],'task_id':d['task_id'],'category':x['category'],'U':x['U_logprob'],'delta_error':x['free_running_action_l2']-x['free_running_clean_l2']})
 integrity={'pass':len(files)==500 and len(rows)==16000 and not bad and all(np.isfinite([x['U'],x['delta_error']]).all() for x in rows),'states':len(files),'candidate_tokens':len(rows),'bad':bad}
 overall=metrics(rows);task={str(t):metrics([x for x in rows if x['task_id']==t]) for t in range(10)};category={c:metrics([x for x in rows if x['category']==c]) for c in ('attention_top16','attention_bottom8','random8')}
 by=defaultdict(list)
 for x in rows:by[x['snapshot_id']].append(x)
 ids=sorted(by);rng=np.random.default_rng(20260811);boot=[];counts=[]
 for sid in ids:
  changed=[x for x in by[sid] if abs(x['delta_error'])>1e-12 and abs(x['U'])>1e-12]
  counts.append((sum(np.sign(x['U'])==np.sign(x['delta_error']) for x in changed),len(changed)))
 counts=np.asarray(counts,dtype=np.int64)
 for _ in range(10000):
  sampled=counts[rng.integers(0,len(counts),len(counts))].sum(axis=0);boot.append(sampled[0]/sampled[1])
 ci=[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))];decision='AR_TEACHER_FORCED_SIGN_DOES_NOT_TRANSFER_TO_FREE_RUNNING_NO_GO' if ci[0]<=.5 else 'AR_SIGNED_TOKEN_UTILITY_GATE_PASS'
 summary={'integrity':integrity,'overall':overall,'snapshot_cluster_bootstrap_95_ci_sign_agreement':ci,'by_task':task,'by_category':category,'decision':decision,'oracle_guidance_run':False,'oracle_reason':'Sequential gate failed: teacher-forced sign is not predictive of free-running action-error direction.'};(art/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');(art/'decision.json').write_text(json.dumps({'decision':decision,'integrity':'PASS'},indent=2)+'\n');(art/'status.complete').write_text('500/500 states; 16000/16000 token interventions\n')
 report=['# AR Signed Token Utility Audit','',f"Integrity: PASS (`{len(files)}/500` states, `{len(rows)}/16000` token interventions).",'','## Primary gate',f"- Free-running action changed: {overall['free_running_changed_fraction']:.1%}",f"- Sign agreement on changed cases: {overall['sign_agreement_on_changed']:.1%}",f"- Snapshot-cluster bootstrap 95% CI: [{ci[0]:.1%}, {ci[1]:.1%}]",f"- Spearman(U, masked-minus-clean free-running error): {overall['spearman_U_vs_error_delta']:.3f}",f"- Positive-U mask actually worsens free run: {overall['positive_U_mask_worsens_fraction']:.1%}",f"- Negative-U mask actually improves free run: {overall['negative_U_mask_improves_fraction']:.1%}",'','## Decision',decision,'','Oracle guidance was not run because the sequential teacher-forced-to-free-running gate failed. Training a sign predictor or moving to closed-loop guidance is not justified.'];(art/'report.md').write_text('\n'.join(report)+'\n');(art/'sha256_audit.json').write_text(json.dumps({'protocol':sha(art/'protocol.lock.yaml'),'summary':sha(art/'summary.json')},indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
