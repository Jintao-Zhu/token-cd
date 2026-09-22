#!/usr/bin/env python3
"""Episode-aligned lambda transition and rescue/harm intersection analysis."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


def group_original(arm, vanilla, keys):
    groups=defaultdict(list)
    for k in keys:
        a=bool(arm[k]['success']); v=bool(vanilla[k]['success'])
        if a and not v: groups['original_rescue'].append(k)
        elif not a and v: groups['original_harm'].append(k)
        elif a and v: groups['both_success'].append(k)
        else: groups['both_fail'].append(k)
    return groups


def transition_table(low, arm_base, vanilla, keys):
    groups=group_original(arm_base, vanilla, keys)
    out={}
    for g,ks in groups.items():
        succ=[k for k in ks if bool(low[k]['success'])]
        fail=[k for k in ks if not bool(low[k]['success'])]
        out[g]={'n':len(ks),'low_success':len(succ),'low_fail':len(fail),'low_success_rate':len(succ)/len(ks) if ks else None}
    return out


def intersection_table(a, b, vanilla, keys, group):
    if group=='vanilla_fail':
        ks=[k for k in keys if not bool(vanilla[k]['success'])]
        a_s=set(k for k in ks if bool(a[k]['success']))
        b_s=set(k for k in ks if bool(b[k]['success']))
        return {
            'n':len(ks),'both_rescue':len(a_s&b_s),'only_a_rescue':len(a_s-b_s),
            'only_b_rescue':len(b_s-a_s),'neither':len(set(ks)-a_s-b_s),
        }
    if group=='vanilla_success':
        ks=[k for k in keys if bool(vanilla[k]['success'])]
        a_h=set(k for k in ks if not bool(a[k]['success']))
        b_h=set(k for k in ks if not bool(b[k]['success']))
        return {
            'n':len(ks),'both_harm':len(a_h&b_h),'only_a_harm':len(a_h-b_h),
            'only_b_harm':len(b_h-a_h),'both_preserved':len(set(ks)-a_h-b_h),
        }
    raise ValueError(group)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=Path('artifacts/libero_lambda_transition_analysis_v1'))
    a=p.parse_args()
    roots={
        'vanilla':Path('artifacts/libero_official_vanilla_500_v1'),
        'current_lam050':Path('artifacts/libero_official_matched_500_v1'),
        'current_lam025':Path('artifacts/libero_lambda_sweep_v1/current_lam025'),
        'current_lam050sweep':Path('artifacts/libero_lambda_sweep_v1/current_lam050'),
        'selected5_lam050':Path('artifacts/libero_ranking_repair_matrix_v1/selected5_endpoint'),
        'selected5_lam025':Path('artifacts/libero_lambda_sweep_v1/selected5_lam025'),
        'selected5_lam050sweep':Path('artifacts/libero_lambda_sweep_v1/selected5_lam050'),
    }
    data={k:load(v) for k,v in roots.items()}
    keys=sorted(set.intersection(*(set(v) for v in data.values())))
    out={'n':len(keys)}
    # current λ0.5 groups; evaluate λ0.25 and λ0.125
    out['current_transitions']={
        'lambda_0.25': transition_table(data['current_lam050sweep'], data['current_lam050'], data['vanilla'], keys),
        'lambda_0.125': transition_table(data['current_lam025'], data['current_lam050'], data['vanilla'], keys),
    }
    out['selected5_transitions']={
        'lambda_0.25': transition_table(data['selected5_lam050sweep'], data['selected5_lam050'], data['vanilla'], keys),
        'lambda_0.125': transition_table(data['selected5_lam025'], data['selected5_lam050'], data['vanilla'], keys),
    }
    # best configs intersection
    out['best_intersections']={
        'vanilla_fail': intersection_table(data['current_lam025'], data['selected5_lam050'], data['vanilla'], keys, 'vanilla_fail'),
        'vanilla_success': intersection_table(data['current_lam025'], data['selected5_lam050'], data['vanilla'], keys, 'vanilla_success'),
    }
    # per-task best intersections
    out['tasks']={}
    for task in sorted({k[0] for k in keys}):
        ks=[k for k in keys if k[0]==task]
        out['tasks'][task]={
            'vanilla_fail': intersection_table(data['current_lam025'], data['selected5_lam050'], data['vanilla'], ks, 'vanilla_fail'),
            'vanilla_success': intersection_table(data['current_lam025'], data['selected5_lam050'], data['vanilla'], ks, 'vanilla_success'),
        }
    outdir=a.out.resolve(); outdir.mkdir(parents=True,exist_ok=True)
    (outdir/'TRANSITION_SUMMARY.json').write_text(json.dumps(out,indent=2)+'\n')
    lines=['# Lambda transition / rescue-harm intersection analysis','',f'paired={len(keys)}','','## current: base λ0.5 vs vanilla, then lower λ','']
    for lam,rec in out['current_transitions'].items():
        lines.append(f'### current -> {lam}')
        lines += ['| original group | n | low success | low fail | rate |','|---|---:|---:|---:|---:|']
        for g,r in rec.items(): lines.append(f"| {g} | {r['n']} | {r['low_success']} | {r['low_fail']} | {r['low_success_rate']:.3f} |")
        lines.append('')
    lines += ['## selected5: base λ0.5 vs vanilla, then lower λ','']
    for lam,rec in out['selected5_transitions'].items():
        lines.append(f'### selected5 -> {lam}')
        lines += ['| original group | n | low success | low fail | rate |','|---|---:|---:|---:|---:|']
        for g,r in rec.items(): lines.append(f"| {g} | {r['n']} | {r['low_success']} | {r['low_fail']} | {r['low_success_rate']:.3f} |")
        lines.append('')
    lines += ['## Best-config intersections: current λ0.125 vs selected5 λ0.5','']
    for group,rec in out['best_intersections'].items():
        lines.append(f'### {group}')
        lines.append(json.dumps(rec,indent=2))
        lines.append('')
    (outdir/'TRANSITION_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:40]))
    print('saved',outdir)

if __name__=='__main__': main()
