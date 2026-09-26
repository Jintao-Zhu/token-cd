#!/usr/bin/env python3
"""Episode-cluster analysis for L11 Rank-Causal Effect offline traces."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def average_ranks(values):
    values=np.asarray(values,float); order=np.argsort(values,kind='mergesort'); ranks=np.empty(len(values),float)
    i=0
    while i<len(values):
        j=i+1
        while j<len(values) and values[order[j]]==values[order[i]]: j+=1
        ranks[order[i:j]]=(i+1+j)/2.0
        i=j
    return ranks


def spearman(x,y):
    rx,ry=average_ranks(x),average_ranks(y)
    if np.ptp(ry)==0:return 0.0
    return float(np.corrcoef(rx,ry)[0,1])


def load_episodes(root: Path, benchmark: str):
    files = (sorted(root.glob('episodes/*/seed_*_rank_effects.json')) if benchmark == 'simpler'
             else sorted(root.glob('episodes/*/rank_effects.json')))
    out=[]
    for path in files:
        d=json.loads(path.read_text())
        trace=d.get('rank_effect_trace', [])
        if benchmark == 'libero' and d.get('integrity',{}).get('replay_mismatches') != 0:
            continue
        state_curves=[]
        for state in trace:
            bins=state.get('rank_bins')
            if not bins or len(bins)!=8 or [b.get('rank_bin') for b in bins] != list(range(1,9)):
                continue
            vals=[float(b['effect_score']) for b in bins]
            if not np.isfinite(vals).all():
                continue
            state_curves.append(vals)
        if not state_curves:
            continue
        curve=np.asarray(state_curves,dtype=float).mean(axis=0)
        flat=bool(np.ptp(curve)==0)
        rho=0.0 if flat else spearman(-np.arange(1,9),curve)
        task=d.get('task') if benchmark=='simpler' else d.get('task_name')
        out.append({'path':str(path),'task':task,'episode_id':path.parent.name if benchmark=='libero' else f"{task}/seed_{d['seed']:03d}",
                    'seed':d.get('seed',d.get('init_state_id')),'n_states':len(state_curves),
                    'success':d.get('source_success'),'curve':curve.tolist(),'rho':rho,'flat_curve':flat})
    return out


def summarize(rows, rng, bootstrap_n=10000):
    if not rows: return {'n_episodes':0}
    X=np.asarray([x['curve'] for x in rows],float)
    diffs=X[:,0]-X[:,7]
    rhos=np.asarray([x['rho'] for x in rows],float)
    n=len(rows)
    draw=rng.integers(0,n,size=(bootstrap_n,n))
    boot_diff=diffs[draw].mean(axis=1)
    boot_rho=rhos[draw].mean(axis=1)
    return {'n_episodes':n,'n_states':int(sum(x['n_states'] for x in rows)),
            'mean_effect_by_bin':X.mean(axis=0).tolist(),
            'mean_B1_minus_B8':float(diffs.mean()),
            'B1_minus_B8_bootstrap_95ci':[float(x) for x in np.quantile(boot_diff,[.025,.975])],
            'mean_episode_spearman':float(rhos.mean()),
            'episode_spearman_bootstrap_95ci':[float(x) for x in np.quantile(boot_rho,[.025,.975])],
            'median_episode_spearman':float(np.median(rhos)),
            'positive_gradient_fraction':float(np.mean(rhos>0)),
            'flat_curve_fraction':float(np.mean([x['flat_curve'] for x in rows]))}



def summarize_benchmark(rows, rng, bootstrap_n=10000):
    """Equal-task macro with within-task episode-cluster bootstrap."""
    if not rows: return {'n_episodes':0}
    by_task={task:[x for x in rows if x['task']==task] for task in sorted({x['task'] for x in rows})}
    curves={task:np.asarray([x['curve'] for x in group],float) for task,group in by_task.items()}
    rhos={task:np.asarray([x['rho'] for x in group],float) for task,group in by_task.items()}
    task_means=np.stack([curves[t].mean(axis=0) for t in by_task])
    macro_curve=task_means.mean(axis=0)
    task_diffs=np.asarray([curves[t][:,0].mean()-curves[t][:,7].mean() for t in by_task])
    macro_rho=float(np.mean([rhos[t].mean() for t in by_task]))
    boot_d=[]; boot_r=[]
    for _ in range(bootstrap_n):
        ds=[]; rs=[]
        for task in by_task:
            n=len(curves[task]); idx=rng.integers(0,n,size=n)
            ds.append(float(np.mean(curves[task][idx,0]-curves[task][idx,7])))
            rs.append(float(np.mean(rhos[task][idx])) )
        boot_d.append(float(np.mean(ds))); boot_r.append(float(np.mean(rs)))
    all_rhos=np.concatenate(list(rhos.values()))
    return {'n_episodes':len(rows),'n_states':int(sum(x['n_states'] for x in rows)),
            'aggregation':'equal-task macro; bootstrap resamples episodes within each task',
            'episodes_per_task':{t:len(g) for t,g in by_task.items()},
            'mean_effect_by_bin':macro_curve.tolist(),
            'mean_B1_minus_B8':float(task_diffs.mean()),
            'B1_minus_B8_stratified_episode_bootstrap_95ci':[float(x) for x in np.quantile(boot_d,[.025,.975])],
            'mean_episode_spearman':macro_rho,
            'episode_spearman_stratified_bootstrap_95ci':[float(x) for x in np.quantile(boot_r,[.025,.975])],
            'median_episode_spearman':float(np.median(all_rhos)),
            'positive_gradient_fraction':float(np.mean(all_rhos>0)),
            'flat_curve_fraction':float(np.mean([x['flat_curve'] for x in rows]))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--artifact',type=Path,required=True); ap.add_argument('--bootstrap',type=int,default=10000); a=ap.parse_args()
    rng=np.random.default_rng(20260926); result={'protocol_id':'L11_RANK_CAUSAL_EFFECT_OFFLINE_V1','bootstrap_unit':'episode','bootstrap_seed':20260926}
    for name in ('simpler','libero'):
        rows=load_episodes(a.artifact/f'offline_{name}',name)
        result[name]={'overall':summarize_benchmark(rows,rng,a.bootstrap),'tasks':{}}
        for task in sorted({x['task'] for x in rows}):
            result[name]['tasks'][task]=summarize([x for x in rows if x['task']==task],rng,a.bootstrap)
        result[name]['episodes']=rows
    out=a.artifact/'offline_analysis.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    for name in ('simpler','libero'):
        print(name,json.dumps(result[name]['overall'],sort_keys=True))
        for task,summary in result[name]['tasks'].items(): print(' ',task,json.dumps(summary,sort_keys=True))
    print('wrote',out)
if __name__=='__main__': main()
