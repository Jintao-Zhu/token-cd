#!/usr/bin/env python3
"""Analyze paired rescue/harm episodes for selected5_endpoint vs vanilla."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load(root: Path, min_state=20, max_state=49):
    rows={}
    for path in root.glob('*/episode_*.json'):
        row=json.loads(path.read_text())
        state=int(row.get('init_state_index',row['episode']))
        if min_state<=state<=max_state:
            rows[(row['task'],state)]=row
    return rows


def first_divergence(a,b,atol=1e-8):
    ta=a.get('trajectory',[]); tb=b.get('trajectory',[])
    for i,(x,y) in enumerate(zip(ta,tb)):
        if not np.allclose(np.asarray(x),np.asarray(y),rtol=0,atol=atol):
            return i,float(np.linalg.norm(np.asarray(x)-np.asarray(y)))
    return min(len(ta),len(tb)),0.0


def trace_mean(row,key):
    vals=[x.get(key) for x in row.get('trace',[]) if x.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def guided_change_mean(row):
    vals=[]
    for x in row.get('trace',[]):
        if x.get('guided_change_ratio') is not None:
            vals.append(float(x['guided_change_ratio']))
        elif x.get('guided_changed_dims') is not None:
            vals.append(float(x['guided_changed_dims']) / 6.0)
    return float(np.mean(vals)) if vals else None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--selected-root',type=Path,default=Path('artifacts/libero_ranking_repair_matrix_v1/selected5_endpoint'))
    p.add_argument('--vanilla-root',type=Path,default=Path('artifacts/libero_official_vanilla_500_v1'))
    p.add_argument('--out',type=Path,default=Path('artifacts/libero_rescue_harm_analysis_v1'))
    a=p.parse_args()
    sel=load(a.selected_root.resolve()); van=load(a.vanilla_root.resolve())
    keys=sorted(set(sel)&set(van)); out=a.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list)
    for k in keys:
        s=bool(sel[k]['success']); v=bool(van[k]['success'])
        if s and not v: name='rescue'
        elif v and not s: name='harm'
        elif s and v: name='both_success'
        else: name='both_fail'
        groups[name].append(k)
    report={'counts':{name:len(v) for name,v in groups.items()},'tasks':{},'groups':{}}
    for name,ks in groups.items():
        rec={'n':len(ks)}
        for label,rowset in [('selected',sel),('vanilla',van)]:
            rec[label+'_steps_mean']=float(np.mean([rowset[k]['steps'] for k in ks])) if ks else None
            rec[label+'_m_mean']=float(np.mean([rowset[k].get('selected_token_count_mean',np.nan) for k in ks])) if ks else None
        if ks:
            rec['selected_feature_perturbation_mean']=float(np.mean([trace_mean(sel[k],'feature_perturbation_relative') or np.nan for k in ks]))
            rec['selected_guided_change_mean']=float(np.mean([guided_change_mean(sel[k]) or np.nan for k in ks]))
            div=[first_divergence(sel[k],van[k]) for k in ks]
            rec['first_divergence_step_mean']=float(np.mean([x[0] for x in div]))
            rec['first_divergence_action_l2_mean']=float(np.mean([x[1] for x in div]))
        report['groups'][name]=rec
    for task in sorted({k[0] for k in keys}):
        row={'counts':{}}
        for name,ks0 in groups.items():
            ks=[k for k in ks0 if k[0]==task]
            row['counts'][name]=len(ks)
            if ks:
                div=[first_divergence(sel[k],van[k]) for k in ks]
                row.setdefault('first_divergence_step_mean',{})[name]=float(np.mean([x[0] for x in div]))
                row.setdefault('selected_m_mean',{})[name]=float(np.mean([sel[k].get('selected_token_count_mean',np.nan) for k in ks]))
        report['tasks'][task]=row
    (out/'RESCUE_HARM_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# selected5 rescue/harm analysis','',f'paired={len(keys)}','','| group | n | first divergence step | action L2 | selected steps | vanilla steps | selected m | feature perturbation | guided change |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ['rescue','harm','both_success','both_fail']:
        r=report['groups'].get(name,{})
        if not r: continue
        lines.append(f"| {name} | {r.get('n',0)} | {r.get('first_divergence_step_mean',float('nan')):.2f} | {r.get('first_divergence_action_l2_mean',float('nan')):.5f} | {r.get('selected_steps_mean',float('nan')):.2f} | {r.get('vanilla_steps_mean',float('nan')):.2f} | {r.get('selected_m_mean',float('nan')):.2f} | {r.get('selected_feature_perturbation_mean',float('nan')):.5f} | {r.get('selected_guided_change_mean',float('nan')):.4f} |")
    (out/'RESCUE_HARM_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__': main()
