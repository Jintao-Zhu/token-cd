#!/usr/bin/env python3
"""Apply the LIBERO episode-level action-value definition to stored SIMPLER logits."""
from __future__ import annotations
import csv, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path('/home/leju-suzhou/zjt_ws/token-cd')
ART=ROOT/'artifacts/prompt_attn_l11_matched_full_9task_0_299_v1'
OUT=ROOT/'artifacts/simpler_action_value_transfer_analysis_v1'
FOUR={'google_robot_open_drawer','google_robot_close_drawer',
      'google_robot_pick_coke_can','google_robot_move_near'}


def episode_metric(path:Path):
    z=np.load(path)
    pos=z['positive'].astype(np.float64)
    neg=z['negative'].astype(np.float64)
    if pos.ndim!=3 or neg.shape!=pos.shape or pos.shape[1]<6:
        raise ValueError(f'bad logits shape {path}: {pos.shape}/{neg.shape}')
    clean=[]; guidance=[]
    for t in range(pos.shape[0]):
        for q in range(6):
            order=np.argsort(pos[t,q])[::-1]
            a1,a2=int(order[0]),int(order[1])
            clean.append(float(pos[t,q,a1]-pos[t,q,a2]))
            guidance.append(float((pos[t,q,a1]-neg[t,q,a1])-(pos[t,q,a2]-neg[t,q,a2])))
    return {'median_G':float(np.median(guidance)),
        'median_clean_margin':float(np.median(clean)),
        'fraction_G_negative':float(np.mean(np.asarray(guidance)<0)),
        'steps':int(pos.shape[0]),'stored_logit_dtype':str(z['positive'].dtype)}


def auc(pos,neg):
    a=np.asarray(pos); b=np.asarray(neg)
    return float((np.sum(a[:,None]>b[None,:])+0.5*np.sum(a[:,None]==b[None,:]))/(len(a)*len(b)))


def summarize(name, cases, bootstrap_seed):
    rescue=[c for c in cases if c['outcome']=='rescue']
    harm=[c for c in cases if c['outcome']=='harm']
    if not rescue or not harm: raise ValueError(f'{name}: missing class rescue={len(rescue)} harm={len(harm)}')
    r=[c['median_G'] for c in rescue]; h=[c['median_G'] for c in harm]
    rng=np.random.default_rng(bootstrap_seed)
    draws=np.empty(20000)
    for i in range(len(draws)):
        draws[i]=auc(rng.choice(r,len(r),replace=True),rng.choice(h,len(h),replace=True))
    per_task={}
    for task in sorted({c['task'] for c in cases}):
        rr=[c['median_G'] for c in cases if c['task']==task and c['outcome']=='rescue']
        hh=[c['median_G'] for c in cases if c['task']==task and c['outcome']=='harm']
        per_task[task]={'rescue_n':len(rr),'harm_n':len(hh),
            'rescue_median_G':float(np.median(rr)) if rr else None,
            'harm_median_G':float(np.median(hh)) if hh else None,
            'auc_rescue_higher_G':auc(rr,hh) if rr and hh else None}
    return {'n_pairs_in_cohort':len(cases),'rescue_n':len(rescue),'harm_n':len(harm),
       'rescue_median_G':float(np.median(r)),'harm_median_G':float(np.median(h)),
       'median_G_difference_rescue_minus_harm':float(np.median(r)-np.median(h)),
       'auc_rescue_higher_G':auc(r,h),
       'episode_bootstrap_95_percentile_ci':[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))],
       'rescue_episode_median_fraction_G_negative':float(np.median([c['fraction_G_negative'] for c in rescue])),
       'harm_episode_median_fraction_G_negative':float(np.median([c['fraction_G_negative'] for c in harm])),
       'rescue_episode_median_clean_margin':float(np.median([c['median_clean_margin'] for c in rescue])),
       'harm_episode_median_clean_margin':float(np.median([c['median_clean_margin'] for c in harm])),
       'per_task':per_task}


def main():
    with (ART/'paired_results_0_299.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    all_discordant=[]; missing=[]; seen=Counter()
    for r in rows:
        v=r['vanilla'].strip().lower()=='true'; m=r['l11_matched'].strip().lower()=='true'
        if v==m: continue
        task=r['task']; seed=int(r['seed'])
        outcome='rescue' if m and not v else 'harm'
        seen[(task,seed)]=1
        path=ART/'run'/'episodes'/task/'prompt_single'/f'episode_{seed}_arrays.npz'
        if not path.exists():
            missing.append({'task':task,'seed':seed,'outcome':outcome})
            continue
        all_discordant.append({'task':task,'seed':seed,'outcome':outcome,
            'vanilla_success':v,'l11_matched_success':m,**episode_metric(path)})
    primary=[c for c in all_discordant if c['task'] in FOUR and 100<=c['seed']<=199]
    secondary=[c for c in all_discordant if 100<=c['seed']<=299]
    # Validate full paired outcome counts directly from the source CSV.
    cohort_counts={}
    for name,pred in {
      'four_google_tasks_seed100_199':lambda r:r['task'] in FOUR and 100<=int(r['seed'])<=199,
      'all_nine_tasks_seed100_299':lambda r:100<=int(r['seed'])<=299,
      'all_nine_tasks_seed0_299':lambda r:True}.items():
        selected=[r for r in rows if pred(r)]
        cc=Counter()
        for r in selected:
            v=r['vanilla'].strip().lower()=='true';m=r['l11_matched'].strip().lower()=='true'
            cc['vanilla_success']+=int(v);cc['l11_success']+=int(m)
            cc['rescue']+=int(m and not v);cc['harm']+=int(v and not m)
        cohort_counts[name]={'pairs':len(selected),**dict(cc)}
    summary={'source_artifact':str(ART),'source_report':str(ART/'FULL_REPORT_0_299.md'),
      'source_protocol':json.loads((ART/'run'/'CONFIG_LOCK.json').read_text())['protocol_id'],
      'metric':'per episode: median over online replan steps and action dimensions 0..5 of G=(z+-z-)(clean top1)-(z+-z-)(clean top2); episode is the independent unit',
      'source_outcome_counts':cohort_counts,
      'primary_four_task_seed100_199':summarize('primary',primary,20260925),
      'secondary_all_tasks_seed100_299':summarize('secondary',secondary,20260926),
      'logit_trace_coverage':{'available_discordant_arrays':len(all_discordant),'missing_discordant_arrays':len(missing),
        'missing_by_outcome':dict(Counter(x['outcome'] for x in missing)),
        'available_by_outcome':dict(Counter(x['outcome'] for x in all_discordant))},
      'limits':['historical SIMPLER and new LIBERO use different task suite/checkpoint and renderer; descriptive comparison does not isolate benchmark as a cause',
        'SIMPLER arrays store float16 logits, LIBERO OSMesa arrays store float32 logits',
        'seed 0..99 SIMPLER logits are not retained in this artifact; they are excluded from trace-level analysis',
        'within-episode steps/dimensions are aggregated; no step is treated as an independent sample'],
      'episode_values':all_discordant,'missing_episode_arrays':missing}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'SIMPLER_ACTION_VALUE_ANALYSIS.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('episode_values','missing_episode_arrays')},indent=2))

if __name__=='__main__': main()
