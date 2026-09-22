#!/usr/bin/env python3
"""Aggregate matched-only LIBERO-Spatial results."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument('--artifact',type=Path,required=True); a=p.parse_args()
    root=a.artifact.resolve(); registry=json.loads(Path('artifacts/libero/task_registry.json').read_text())['spatial']
    rows=[]
    for task in registry:
        files=sorted((root/task).glob('episode_*.json'))
        for f in files:
            d=json.loads(f.read_text())
            rows.append({'task':task,'episode':d['episode'],'success':bool(d['success']),
                         'steps':int(d['steps']),'mean_m':float(d['selected_token_count_mean'])})
    out={'n':len(rows),'overall_success':int(sum(r['success'] for r in rows)),
         'overall_rate':float(np.mean([r['success'] for r in rows])) if rows else None,
         'per_task':{}}
    lines=['# LIBERO-Spatial matched-only','', '| Task | n | Success | Rate | mean steps | mean m |','|---|---:|---:|---:|---:|---:|']
    for task in registry:
        sub=[r for r in rows if r['task']==task]
        if not sub: continue
        rec={'n':len(sub),'success':int(sum(r['success'] for r in sub)),
             'rate':float(np.mean([r['success'] for r in sub])),
             'mean_steps':float(np.mean([r['steps'] for r in sub])),
             'mean_m':float(np.mean([r['mean_m'] for r in sub]))}
        out['per_task'][task]=rec
        lines.append(f"| {task} | {rec['n']} | {rec['success']} | {rec['rate']:.3f} | {rec['mean_steps']:.1f} | {rec['mean_m']:.2f} |")
    lines += ['', f"**OVERALL**: {out['overall_success']}/{out['n']} = {out['overall_rate']:.3f}"]
    (root/'SUMMARY.json').write_text(json.dumps(out,indent=2)+'\n')
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
if __name__=='__main__': main()
