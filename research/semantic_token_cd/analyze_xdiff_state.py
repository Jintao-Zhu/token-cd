"""Summarize Stage-2 identical-state selector -> reconstruction -> action effects."""
from __future__ import annotations
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from research.semantic_token_cd.xdiff_protocol import ARTIFACT, TASKS, atomic_json

CONFIGS=('semantic_0p5','semantic_1p0','paraphrase_0p5')

def jac(a,b):
    a,b=set(a),set(b); return len(a&b)/max(1,len(a|b))

def main():
    rows=[]; bad=[]
    for p in sorted((ARTIFACT/'stage2_action/states').glob('**/step_*.json')):
        d=json.loads(p.read_text()); audit=d['metrics']['audit']
        if not all(audit.values()): bad.append(str(p))
        base=d['metrics']['correct']; bt=np.asarray(base['final_token_ids'][:6]); ba=np.asarray(base['guided_action'][:6])
        for name in CONFIGS:
            x=d['metrics'][name]; xt=np.asarray(x['final_token_ids'][:6]); xa=np.asarray(x['guided_action'][:6])
            rows.append({'task':d['task'],'seed':d['seed'],'control_step':d['control_step'],'config':name,'m':x['m'],
              'mask_jaccard':jac(base['selected_token_ids'],x['selected_token_ids']),
              'replaced_tokens':len(set(x['selected_token_ids'])-set(base['selected_token_ids'])),
              'residual_cosine':x['residual_cosine_vs_correct'],
              'residual_norm_ratio':x['centered_residual_norm']/max(1e-12,base['centered_residual_norm']),
              'feature_norm_ratio':x['feature_perturbation_norm']/max(1e-12,base['feature_perturbation_norm']),
              'action_exact':bool(np.array_equal(xt,bt)), 'changed_dims':int(np.sum(xt!=bt)),
              'action_l2_vs_correct':float(np.linalg.norm(xa-ba)),
              **{f'dim_{i}_changed':bool(xt[i]!=bt[i]) for i in range(6)}})
    if bad or len(rows)!=240*len(CONFIGS): raise RuntimeError(f'incomplete stage2 rows={len(rows)} bad={bad[:3]}')
    stats=ARTIFACT/'stage2_action/statistics'; stats.mkdir(parents=True,exist_ok=True)
    with (stats/'state_action_metrics.csv').open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary={'n_states':240,'audits_passed':True,'overall':{},'by_task':{}}
    def aggregate(rr):
        return {'n':len(rr),'mean_mask_jaccard':float(np.mean([r['mask_jaccard'] for r in rr])),
          'mean_replaced_tokens':float(np.mean([r['replaced_tokens'] for r in rr])),
          'action_exact_fraction':float(np.mean([r['action_exact'] for r in rr])),
          'mean_changed_dims':float(np.mean([r['changed_dims'] for r in rr])),
          'mean_action_l2_vs_correct':float(np.mean([r['action_l2_vs_correct'] for r in rr])),
          'mean_residual_cosine':float(np.mean([r['residual_cosine'] for r in rr])),
          'mean_residual_norm_ratio':float(np.mean([r['residual_norm_ratio'] for r in rr])),
          'residual_over_2x_fraction':float(np.mean([r['residual_norm_ratio']>2 for r in rr])),
          'mean_feature_norm_ratio':float(np.mean([r['feature_norm_ratio'] for r in rr])),
          'per_dim_change_fraction':[float(np.mean([r[f'dim_{i}_changed'] for r in rr])) for i in range(6)]}
    for name in CONFIGS: summary['overall'][name]=aggregate([r for r in rows if r['config']==name])
    for task in TASKS:
        summary['by_task'][task]={name:aggregate([r for r in rows if r['task']==task and r['config']==name]) for name in CONFIGS}
    atomic_json(stats/'summary.json',summary); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
