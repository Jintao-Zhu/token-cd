#!/usr/bin/env python3
"""Audit and summarize action-value traces from the OSMesa LIBERO pilot."""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np


def summarize_arm(root: Path, case_id: str, arm: str, pair_outcome: str):
    ep_path=root/'episodes'/case_id/arm/'episode.json'
    ep=json.loads(ep_path.read_text())
    arrays=np.load(ep_path.parent/'step_arrays.npz')
    zp=arrays['positive_action_logits'].astype(np.float64)
    zn=arrays['negative_action_logits'].astype(np.float64)
    zg=arrays['guided_action_logits'].astype(np.float64)
    trace=ep['trace']
    steps=len(trace)
    qn=min(6,zp.shape[1])
    row_metrics=[]
    per_dim=defaultdict(list)
    per_phase=defaultdict(list)
    first_change=None
    changed_replans=0
    max_consecutive=run=0
    for t,row in enumerate(trace):
        changed=int(row['clean_vs_guided_changed_dims'])
        changed_replans += changed>0
        run=run+1 if changed>0 else 0
        max_consecutive=max(max_consecutive,run)
        if changed>0 and first_change is None: first_change=t
        for q in range(qn):
            z0=zp[t,q]; zneg=zn[t,q]; zfinal=zg[t,q]
            order=np.argsort(z0)[::-1]
            a1,a2=int(order[0]),int(order[1])
            margin=float(z0[a1]-z0[a2])
            g=float((z0[a1]-zneg[a1])-(z0[a2]-zneg[a2]))
            guided_margin=float(zfinal[a1]-zfinal[a2])
            flip=bool(int(np.argmax(zfinal))!=a1)
            phase=row['task_phase']
            rec={'step':t,'dim':q,'phase':phase,'clean_margin':margin,'G':g,
                 'predicted_guided_margin':guided_margin,'winner_flip':flip,
                 'selected_m':int(row['matched_m']), 'changed_dims':changed}
            row_metrics.append(rec); per_dim[q].append(rec); per_phase[phase].append(rec)
    if not row_metrics:
        raise ValueError(f'No action-dimension metrics: {case_id}/{arm}')
    def metrics(rs):
        return {
          'n_dim_steps':len(rs),
          'clean_margin_median':float(np.median([r['clean_margin'] for r in rs])),
          'guidance_G_median':float(np.median([r['G'] for r in rs])),
          'guidance_G_negative_fraction':float(np.mean([r['G']<0 for r in rs])),
          'predicted_guided_margin_median':float(np.median([r['predicted_guided_margin'] for r in rs])),
          'clean_winner_flip_rate':float(np.mean([r['winner_flip'] for r in rs])),
          'high_margin_flip_n':int(sum(r['winner_flip'] and r['clean_margin']>=global_q75 for r in rs)),
          'high_margin_flip_rate_of_all_steps':float(np.mean([r['winner_flip'] and r['clean_margin']>=global_q75 for r in rs])),
        }
    summary={
      'case_id':case_id,'arm':arm,'pair_outcome':pair_outcome,'task_id':ep['task_id'],
      'task_name':ep['task_name'],'success':bool(ep['success']),'steps':steps,
      'first_stable_grasp_step':ep.get('first_stable_grasp_step'),
      'first_intervened_replan':first_change,
      'first_intervened_normalized':float(first_change/steps) if first_change is not None and steps else None,
      'changed_replans':int(changed_replans),'changed_replan_rate':float(changed_replans/steps) if steps else 0.0,
      'max_consecutive_changed_replans':int(max_consecutive),
      'overall':metrics(row_metrics),
      'per_action_dim':{str(q):metrics(rs) for q,rs in sorted(per_dim.items())},
      'per_phase':{ph:metrics(rs) for ph,rs in sorted(per_phase.items())},
    }
    # Actual decoded action displacement is useful alongside logit margins.
    deltas=[]
    for row in trace:
        clean=np.asarray(row['clean_action_from_positive_logits'],dtype=float)
        guided=np.asarray(row['guided_action'],dtype=float)
        deltas.append(guided-clean)
    delta=np.asarray(deltas)
    summary['decoded_action_delta']={
      'l2_median':float(np.median(np.linalg.norm(delta[:,:6],axis=1))),
      'per_dim_abs_median':[float(x) for x in np.median(np.abs(delta[:,:6]),axis=0)],
      'per_dim_signed_median':[float(x) for x in np.median(delta[:,:6],axis=0)],
      'gripper_changed_rate':float(np.mean(np.abs(delta[:,6])>1e-8)) if delta.shape[1]>6 else None,
    }
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--artifact',type=Path,required=True)
    a=p.parse_args(); root=a.artifact.resolve()
    audit=json.loads((root/'FINAL_AUDIT.json').read_text())
    if audit['summary']['status']!='PASS': raise SystemExit('Refusing analysis: final audit is not PASS')
    cases=json.loads((root/'cases_manifest.json').read_text())['cases']
    pairs=[json.loads((root/'pairs'/f"{c['case_id']}.json").read_text()) for c in cases]
    global_margins=[]
    for pair in pairs:
        cid=pair['case_id']
        ep=json.loads((root/'episodes'/cid/'matched'/'episode.json').read_text())
        z=np.load(root/'episodes'/cid/'matched'/'step_arrays.npz')['positive_action_logits']
        for t in range(z.shape[0]):
            for q in range(min(6,z.shape[1])):
                ordered=np.partition(z[t,q],-2)[-2:]
                global_margins.append(float(ordered.max()-ordered.min()))
    global_q75=float(np.quantile(global_margins,0.75))
    globals()['global_q75']=global_q75
    by_case=[]
    for pair in pairs:
        by_case.append({'pair':pair,'arms':{
            arm:summarize_arm(root,pair['case_id'],arm,pair['outcome'])
            for arm in ('vanilla','matched')}})
    outcomes=Counter(p['outcome'] for p in pairs)
    groups={}
    for outcome in ('rescue','harm','concordant'):
        selected=[x for x in by_case if x['pair']['outcome']==outcome]
        arm_groups={}
        for arm in ('vanilla','matched'):
            keys=('clean_margin_median','guidance_G_median','guidance_G_negative_fraction',
                  'predicted_guided_margin_median','clean_winner_flip_rate','high_margin_flip_rate_of_all_steps')
            values={k:[x['arms'][arm]['overall'][k] for x in selected] for k in keys}
            arm_groups[arm]={k:{'n_episodes':len(v),'episode_median':float(np.median(v)) if v else None,
                                'episode_values':v} for k,v in values.items()}
            arm_groups[arm]['changed_replan_rate']={'n_episodes':len(selected),
                'episode_median':float(np.median([x['arms'][arm]['changed_replan_rate'] for x in selected])) if selected else None,
                'episode_values':[x['arms'][arm]['changed_replan_rate'] for x in selected]}
        groups[outcome]={'n_pairs':len(selected),'arms':arm_groups}
    phase_groups={}
    for outcome in ('rescue','harm','concordant'):
        selected=[x for x in by_case if x['pair']['outcome']==outcome]
        phases=sorted(set().union(*(x['arms']['matched']['per_phase'].keys() for x in selected))) if selected else []
        phase_groups[outcome]={}
        for ph in phases:
            episode_values=[]
            for x in selected:
                d=x['arms']['matched']['per_phase'].get(ph)
                if d: episode_values.append(d)
            phase_groups[outcome][ph]={
              'n_episodes_with_phase':len(episode_values),
              'episode_median_clean_margin':float(np.median([v['clean_margin_median'] for v in episode_values])) if episode_values else None,
              'episode_median_guidance_G':float(np.median([v['guidance_G_median'] for v in episode_values])) if episode_values else None,
              'episode_median_winner_flip_rate':float(np.median([v['clean_winner_flip_rate'] for v in episode_values])) if episode_values else None,
            }
    report={'protocol_id':'LIBERO90_TRACE_FAITHFUL_ACTION_VALUE_PILOT_V2_OSMESA',
      'renderer':'osmesa','complete_pairs':len(pairs),'outcomes':dict(outcomes),
      'clean_margin_global_p75_threshold':global_q75,
      'definition':{'M_clean':'z+(a1)-z+(a2), where a1/a2 are clean top-1/top-2 within each of first six action dimensions',
                    'G':'(z+-z-)(a1)-(z+-z-)(a2)',
                    'predicted_guided_margin':'z*(a1)-z*(a2), expected M_clean+0.5*G'},
      'group_summary_episode_level':groups,'matched_arm_phase_summary':phase_groups,
      'per_pair_episode_summaries':by_case,
      'interpretation_limits':['episodes are the independent units; action dimensions and time steps are not independent samples',
        '40 pairs are exploratory; rescue/harm group sizes may be small, especially harm',
        'these findings apply to the OSMesa renderer condition and do not establish why canonical EGL-LIBERO differs from SIMPLER',
        'clean margin is a confidence proxy, not a per-step correctness label'],
      'next_scientific_step':'Use episode-level and task-stratified patterns to update the action-value hypothesis. Do not claim causal stage mechanism from descriptive associations; design the minimal intervention pilot only if a discriminative pattern is supported.'}
    out=root/'ACTION_VALUE_ANALYSIS.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'analysis':str(out),'n_pairs':len(pairs),'outcomes':dict(outcomes),'global_margin_p75':global_q75,'groups':groups,'phase_summary':phase_groups},indent=2))

if __name__=='__main__': main()
