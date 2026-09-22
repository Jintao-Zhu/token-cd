#!/usr/bin/env python3
"""Aggregate the frozen five-task LIBERO-90 paired evaluation."""
from __future__ import annotations
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np


def load_pairs(root: Path):
    pairs=[]
    for p in sorted((root/'pairs').glob('*.json')):
        pairs.append(json.loads(p.read_text()))
    return pairs


def mcnemar_exact(rescue: int, harm: int) -> float:
    n = rescue + harm
    if n == 0:
        return 1.0
    k = min(rescue, harm)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def stats(pairs):
    n=len(pairs)
    v=sum(bool(p['vanilla_success']) for p in pairs)
    m=sum(bool(p['matched_success']) for p in pairs)
    rescue=sum(bool(p['matched_success']) and not bool(p['vanilla_success']) for p in pairs)
    harm=sum(bool(p['vanilla_success']) and not bool(p['matched_success']) for p in pairs)
    return {
        'n':n,'vanilla_success':v,'matched_success':m,
        'vanilla_rate':v/n if n else None,'matched_rate':m/n if n else None,
        'rate_diff':(m-v)/n if n else None,
        'rescue':rescue,'harm':harm,'net':rescue-harm,
        'mcnemar_exact_p':mcnemar_exact(rescue,harm),
    }


def stratified_bootstrap(pairs, seed=20260922, reps=10000):
    rng=np.random.default_rng(seed)
    by_task=defaultdict(list)
    for p in pairs:
        by_task[p['task_name']].append(p)
    tasks=sorted(by_task)
    diffs=[]; nets=[]; matched_rates=[]; vanilla_rates=[]
    for _ in range(reps):
        sample=[]
        for task in tasks:
            group=by_task[task]
            if not group: continue
            idx=rng.integers(0,len(group),size=len(group))
            sample.extend(group[int(i)] for i in idx)
        s=stats(sample)
        diffs.append(s['rate_diff']); nets.append(s['net'])
        matched_rates.append(s['matched_rate']); vanilla_rates.append(s['vanilla_rate'])
    def ci(x):
        return [float(np.quantile(x,0.025)), float(np.quantile(x,0.975))]
    return {
        'reps':reps,
        'rate_diff_95ci':ci(diffs),
        'net_95ci':ci(nets),
        'matched_rate_95ci':ci(matched_rates),
        'vanilla_rate_95ci':ci(vanilla_rates),
    }


def budget_summary(root: Path, pairs):
    values=[]; steps=[]
    for p in pairs:
        if p.get('matched_mean_m') is not None:
            values.append(float(p['matched_mean_m']))
    return {
        'n':len(values),
        'mean':float(np.mean(values)) if values else None,
        'std':float(np.std(values)) if values else None,
        'p10':float(np.quantile(values,0.10)) if values else None,
        'median':float(np.quantile(values,0.50)) if values else None,
        'p90':float(np.quantile(values,0.90)) if values else None,
        'min':float(np.min(values)) if values else None,
        'max':float(np.max(values)) if values else None,
    }


def build_rows(root: Path):
    rows=[]
    for p in sorted((root/'pairs').glob('*.json')):
        d=json.loads(p.read_text())
        rows.append(d)
    return rows


def write_csv(path: Path, rows):
    fields=['case_id','task_id','task_name','instruction','init_state_id','case_seed','init_state_sha256',
            'vanilla_success','matched_success','vanilla_steps','matched_steps','matched_mean_m',
            'vanilla_runtime_seconds','matched_runtime_seconds','worker_id','physical_gpu']
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:r.get(k) for k in fields})


def write_episode_jsonl(root: Path, pairs):
    out=root/'EPISODE_RESULTS.jsonl'
    with out.open('w') as handle:
        for pair in pairs:
            for arm in ('vanilla','matched'):
                p=root/'episodes'/pair['task_name']/arm/f"init_{int(pair['init_state_id']):03d}.json"
                if p.exists():
                    d=json.loads(p.read_text())
                    compact={k:d.get(k) for k in ('protocol_id','case_id','task_id','task_name','arm','init_state_id','init_state_sha256','case_seed','success','normal_end','done','failure_reason','steps','runtime_seconds','checkpoint_revision','code_commit','config_sha256','worker_id','physical_gpu')}
                    if arm=='matched':
                        tr=d.get('trace') or []
                        compact['mean_m']=float(np.mean([x.get('m_effective',0) for x in tr])) if tr else None
                        compact['trace_fields']=['entities','matched_cluster_ids','m_raw','m_effective','unique_mask_count','guided_changed_dims','feature_perturbation_relative','raw_action','executed_action']
                    handle.write(json.dumps(compact,sort_keys=True)+'\n')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--exclude-init-zero',action='store_true'); a=p.parse_args(); root=a.root.resolve()
    pairs=load_pairs(root)
    full=stats(pairs)
    filtered=[x for x in pairs if int(x['init_state_id'])!=0] if a.exclude_init_zero else pairs
    without0=stats(filtered)
    per_task={}
    for task in sorted({x['task_name'] for x in pairs}):
        per_task[task]=stats([x for x in pairs if x['task_name']==task])
    bs=stratified_bootstrap(pairs)
    budget=budget_summary(root,pairs)
    both_success=[x for x in pairs if x['vanilla_success'] and x['matched_success']]
    step_diffs=np.asarray([x['matched_steps']-x['vanilla_steps'] for x in both_success],dtype=float)
    errors=[]
    err_dir=root/'cases'/'error'
    if err_dir.exists():
        for path in sorted(err_dir.glob('*.json')):
            errors.append(json.loads(path.read_text()))
    report={
        'protocol_id':'LIBERO90_FIVE_TASK_SIMPLER_CONFIG_V1',
        'paired_cases_expected':250,
        'paired_cases_available':len(pairs),
        'full':full,
        'excluding_init_state_0':without0,
        'per_task':per_task,
        'bootstrap':bs,
        'matched_budget':budget,
        'both_success_steps':{
            'n':len(both_success),
            'mean_matched_minus_vanilla':float(np.mean(step_diffs)) if len(step_diffs) else None,
            'median_matched_minus_vanilla':float(np.median(step_diffs)) if len(step_diffs) else None,
            'mean_vanilla_steps':float(np.mean([x['vanilla_steps'] for x in both_success])) if both_success else None,
            'mean_matched_steps':float(np.mean([x['matched_steps'] for x in both_success])) if both_success else None,
        },
        'errors':errors,
    }
    (root/'FINAL_RESULTS.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    write_csv(root/'PAIRED_RESULTS.csv',pairs)
    write_episode_jsonl(root,pairs)
    lines=[
        '# LIBERO-90 Five-Task SIMPLER-Config Evaluation','',
        'This is a five-task subset result, not a complete LIBERO-90 score.','',
        '| Task | n | Vanilla | Matched | Rescue | Harm | Net | exact p |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for task,s in per_task.items():
        lines.append(f"| {task} | {s['n']} | {s['vanilla_success']}/{s['n']} | {s['matched_success']}/{s['n']} | {s['rescue']} | {s['harm']} | {s['net']} | {s['mcnemar_exact_p']:.6g} |")
    s=full
    lines += [
        f"| **Overall** | {s['n']} | {s['vanilla_success']}/{s['n']} | {s['matched_success']}/{s['n']} | {s['rescue']} | {s['harm']} | {s['net']} | {s['mcnemar_exact_p']:.6g} |",'',
        f"Overall paired rate difference (matched - vanilla): {s['rate_diff']:.4f}",
        f"Overall bootstrap 95% CI for paired rate difference: [{bs['rate_diff_95ci'][0]:.4f}, {bs['rate_diff_95ci'][1]:.4f}]",
        f"Excluding previously observed init_state 0: n={without0['n']}, vanilla={without0['vanilla_success']}/{without0['n']}, matched={without0['matched_success']}/{without0['n']}, net={without0['net']}, p={without0['mcnemar_exact_p']:.6g}",
        '',
        '## Matched budget distribution',
        '',
        f"- cases with budget: {budget['n']}",
        f"- mean/std: {budget['mean']:.3f} / {budget['std']:.3f}",
        f"- P10/median/P90: {budget['p10']:.3f} / {budget['median']:.3f} / {budget['p90']:.3f}",
        f"- min/max: {budget['min']:.3f} / {budget['max']:.3f}",
        '',
        '## Both-success completion steps',
        '',
        f"- n={report['both_success_steps']['n']}",
        f"- mean matched-minus-vanilla steps: {report['both_success_steps']['mean_matched_minus_vanilla']}",
        f"- mean vanilla/matched steps: {report['both_success_steps']['mean_vanilla_steps']} / {report['both_success_steps']['mean_matched_steps']}",
        '',
        '## Runtime and infrastructure',
        '',
        f"- available paired cases: {len(pairs)} / 250",
        f"- error cases: {len(errors)}",
        f"- errors are listed in FINAL_RESULTS.json and cases/error/",
    ]
    (root/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
