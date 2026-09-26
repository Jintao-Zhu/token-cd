#!/usr/bin/env python3
"""Episode-level paired analysis of harmonic vs feature-kNN negative branches."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT=Path('/home/leju-suzhou/zjt_ws/token-cd')
DATA=ROOT/'artifacts/libero_transfer_diagnostics/negative_variant_probe_v4'
BOOT=20000
SEED=20260927
LAMBDA=.5

def auc(a,b):
    if not len(a) or not len(b): return float('nan')
    d=a[:,None]-b[None,:]
    return float(((d>0).sum()+.5*(d==0).sum())/d.size)

def one_episode(d):
    meta=json.loads((d/'metadata.json').read_text())
    arr=np.load(d/meta['arrays'])
    good=[]
    for i,fr in enumerate(meta['frames']):
        if fr.get('status')!='complete': continue
        idx=int(fr['arrays_index'])
        if idx!=i: # arrays are saved in complete-frame order; metadata can also contain selector misses
            idx=sum(x.get('status')=='complete' for x in meta['frames'][:i])
        zh=arr['positive_harmonic'][idx].astype(np.float32)
        zl=arr['positive_local'][idx].astype(np.float32)
        nh=arr['negative_harmonic'][idx].astype(np.float32)
        nl=arr['negative_local'][idx].astype(np.float32)
        if not (np.array_equal(zh,zl) and np.isfinite(zh[:6]).all() and np.isfinite(nh[:6]).all() and np.isfinite(nl[:6]).all()):
            raise ValueError(f'positive logits mismatch/nonfinite {meta["case_id"]} step {fr["step"]}')
        diffs=[]; flips_h=[]; flips_l=[]; gm_h=[]; gm_l=[]; guided_h=[]; guided_l=[]; neg_cos=[]; neg_rel=[]; neg_argmax_changed=[]
        for q in range(6):
            order=np.argsort(zh[q])[-2:][::-1]
            w,r=map(int,order)
            rh=zh[q]-nh[q]; rl=zh[q]-nl[q]
            an=nh[q].astype(np.float64); bn=nl[q].astype(np.float64)
            neg_cos.append(float(np.dot(an,bn)/(max(np.linalg.norm(an)*np.linalg.norm(bn),1e-12))))
            neg_rel.append(float(np.linalg.norm(an-bn)/max(np.linalg.norm(an),1e-12)))
            neg_argmax_changed.append(float(int(np.argmax(nh[q]))!=int(np.argmax(nl[q]))))
            mh=float(rh[w]-rh[r]); ml=float(rl[w]-rl[r])
            ch=float(zh[q,w]-zh[q,r])
            gh=ch+LAMBDA*mh; gl=ch+LAMBDA*ml
            gm_h.append(mh); gm_l.append(ml); guided_h.append(gh); guided_l.append(gl)
            flips_h.append(float(gh<0)); flips_l.append(float(gl<0))
            diffs.append(ml-mh)
        good.append({'step':int(fr['step']),'harmonic_guidance_margin':float(np.mean(gm_h)),
                     'knn_guidance_margin':float(np.mean(gm_l)),
                     'harmonic_guided_margin':float(np.mean(guided_h)),
                     'knn_guided_margin':float(np.mean(guided_l)),
                     'knn_minus_harmonic_guided_margin':float(np.mean(guided_l)-np.mean(guided_h)),
                     'harmonic_clean_pair_flip_fraction':float(np.mean(flips_h)),
                     'knn_clean_pair_flip_fraction':float(np.mean(flips_l)),
                     'knn_minus_harmonic_flip_fraction':float(np.mean(flips_l)-np.mean(flips_h)),
                     'harmonic_vs_knn_margin_action_dim_sign_agreement':float(np.mean(np.sign(gm_h)==np.sign(gm_l))),
                     'negative_logits_cosine':float(np.mean(neg_cos)),
                     'negative_logits_relative_l2_delta':float(np.mean(neg_rel)),
                     'negative_logits_argmax_changed_fraction':float(np.mean(neg_argmax_changed)),
                     'negative_logits_exactly_equal_fraction':float(np.mean([np.array_equal(nh[q],nl[q]) for q in range(6)]))})
    if not good: return {'case_id':meta['case_id'],'task_id':meta['task_id'],'task':meta['task'],'outcome':meta['outcome'],'valid_states':0,'initial_state_matches_history':meta['initial_state_matches_history'],'replay_success_matches_history':meta['replay_success_matches_history'],'historical_rgb_hash_match_count':meta['historical_rgb_hash_match_count'],'historical_m_and_ids_match_count':meta['historical_m_and_ids_match_count']}
    keys=list(good[0].keys())
    means={k:float(np.mean([x[k] for x in good])) for k in keys if k!='step'}
    return {'case_id':meta['case_id'],'task_id':meta['task_id'],'task':meta['task'],'outcome':meta['outcome'],
            'valid_states':len(good),'initial_state_matches_history':meta['initial_state_matches_history'],
            'replay_success_matches_history':meta['replay_success_matches_history'],
            'historical_rgb_hash_match_count':meta['historical_rgb_hash_match_count'],
            'historical_m_and_ids_match_count':meta['historical_m_and_ids_match_count'],
            'metrics':means,'states':good}

def stat(rows,key):
    valid=[x for x in rows if x.get('valid_states',0)>0 and x['outcome'] in ('rescue','harm')]
    r=np.asarray([x['metrics'][key] for x in valid if x['outcome']=='rescue'])
    h=np.asarray([x['metrics'][key] for x in valid if x['outcome']=='harm'])
    if not len(r) or not len(h): return {'episode_n':len(valid),'rescue_n':len(r),'harm_n':len(h)}
    rng=np.random.default_rng(SEED+sum(map(ord,key)))
    ds=[]; aucs=[]
    for _ in range(BOOT):
        rb=rng.choice(r,len(r),replace=True); hb=rng.choice(h,len(h),replace=True)
        ds.append(float(rb.mean()-hb.mean())); aucs.append(auc(rb,hb))
    tasks=[]
    for tid in sorted({x['task_id'] for x in valid}):
        tr=np.asarray([x['metrics'][key] for x in valid if x['task_id']==tid and x['outcome']=='rescue'])
        th=np.asarray([x['metrics'][key] for x in valid if x['task_id']==tid and x['outcome']=='harm'])
        if len(tr) and len(th): tasks.append({'task_id':tid,'task':next(x['task'] for x in valid if x['task_id']==tid),
                                               'rescue_n':len(tr),'harm_n':len(th),'auc_rescue_higher':auc(tr,th),
                                               'rescue_minus_harm_mean':float(tr.mean()-th.mean())})
    return {'episode_n':len(valid),'rescue_n':len(r),'harm_n':len(h),'rescue_mean':float(r.mean()),'harm_mean':float(h.mean()),
            'rescue_minus_harm_mean':float(r.mean()-h.mean()),'mean_diff_bootstrap_95pct_ci':[float(np.quantile(ds,.025)),float(np.quantile(ds,.975))],
            'auc_rescue_higher':auc(r,h),'auc_bootstrap_95pct_ci':[float(np.quantile(aucs,.025)),float(np.quantile(aucs,.975))],
            'within_task':tasks,'macro_within_task_auc':float(np.mean([x['auc_rescue_higher'] for x in tasks])) if tasks else None}

def main():
    expected=[]
    for name in ['gpu1.txt','gpu3.txt','gpu6.txt','gpu7.txt']:
        expected += (DATA/name).read_text().splitlines()
    missing=[c for c in expected if not ((DATA/c/'metadata.json').is_file() and (DATA/c/'margin_logits.npz').is_file())]
    if missing: raise SystemExit(f'incomplete: {len(missing)} cases: {missing}')
    rows=[one_episode(DATA/c) for c in expected]
    bad=[x['case_id'] for x in rows if not x.get('initial_state_matches_history') or not x.get('replay_success_matches_history')]
    if bad: raise SystemExit(f'replay mismatch: {bad}')
    exact=[x for x in rows if x['valid_states']>0 and x['historical_m_and_ids_match_count']>0]
    metrics=['harmonic_guidance_margin','knn_guidance_margin','knn_minus_harmonic_guided_margin',
             'harmonic_guided_margin','knn_guided_margin','harmonic_clean_pair_flip_fraction',
             'knn_clean_pair_flip_fraction','knn_minus_harmonic_flip_fraction',
             'harmonic_vs_knn_margin_action_dim_sign_agreement','negative_logits_cosine',
             'negative_logits_relative_l2_delta','negative_logits_argmax_changed_fraction','negative_logits_exactly_equal_fraction']
    result={'protocol':'TRACE_VALID_HARMONIC_VS_FEATURE_KNN_NEGATIVE_V1','expected_cases':len(expected),
            'completed_cases':len(rows),'episodes_with_trace_valid_states':len(exact),
            'trace_valid_states':sum(x['valid_states'] for x in exact),
            'outcome_counts':{k:sum(x['outcome']==k for x in exact) for k in ['rescue','harm']},
            'metrics':{k:stat(exact,k) for k in metrics},'episodes':rows,
            'limits':['Episode-level observational comparison on replay states, not a closed-loop causal test.',
                      'Replay starts and outcomes were checked; fresh renderer RGB hashes do not match historical RGB hashes.',
                      'Only states whose current L11 Matched m and selected token IDs equal the historical trace are analyzed.',
                      'Action comparison uses the clean top-two token pair per continuous action dimension and fixed lambda=0.5.',
                      'Feature-kNN changes the harmonic negative construction only; no closed-loop performance claim is supported.']}
    (DATA/'NEGATIVE_VARIANT_PROBE_V4_ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Negative-branch probe v4: harmonic vs feature-kNN','',
           f'Cases completed: {len(rows)}/{len(expected)}. Trace-valid episodes: {len(exact)}; valid states: {result["trace_valid_states"]}.',
           f'Outcomes: {result["outcome_counts"]["rescue"]} Rescue / {result["outcome_counts"]["harm"]} Harm.','',
           'The selector, matched token count, selected IDs, clean logits, RGB frame, and lambda=0.5 were held fixed within each state; only selected-token feature replacement changed. “Guidance margin” is the negative-branch contribution to the clean winner vs runner-up margin.','',
           '| Metric | Rescue | Harm | Rescue−Harm (95% CI) | AUC Rescue higher (95% CI) | Macro within-task AUC |','|---|---:|---:|---:|---:|---:|']
    for k in metrics:
        s=result['metrics'][k]
        if 'rescue_mean' not in s: continue
        ci=s['mean_diff_bootstrap_95pct_ci']; ac=s['auc_bootstrap_95pct_ci']
        lines.append(f'| {k} | {s["rescue_mean"]:.4f} | {s["harm_mean"]:.4f} | {s["rescue_minus_harm_mean"]:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] | {s["auc_rescue_higher"]:.3f} [{ac[0]:.3f}, {ac[1]:.3f}] | {s["macro_within_task_auc"] if s["macro_within_task_auc"] is not None else float("nan"):.3f} |')
    lines+=['','## Interpretation limits','',*[f'- {x}' for x in result['limits']],'']
    (DATA/'NEGATIVE_VARIANT_PROBE_V4_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'cases':len(rows),'trace_valid_episodes':len(exact),'trace_valid_states':result['trace_valid_states'],
                      'outcomes':result['outcome_counts'],'metrics':result['metrics']},indent=2))
if __name__=='__main__': main()
