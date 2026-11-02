"""Select qualitative example panels by the rule frozen in FAILURE_ANNOTATION_PROTOCOL.md.

Rule: two Rescue and two Harm pairs, chosen to be REPRESENTATIVE of the most common
event-derived failure stage in that stratum (not the largest improvements), with a
fixed RNG for tie-breaks. Emits a small JSON index; frame extraction is a separate,
read-only step. Requires the formal membership audit to have passed.
"""
from pathlib import Path
import argparse, json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact', type=Path, required=True)
    ap.add_argument('--stages', default='FORMAL_FAILURE_STAGES.csv')
    a = ap.parse_args()
    art = a.artifact
    audit = json.loads((art / 'FORMAL_MEMBERSHIP_AUDIT.json').read_text())
    if not audit['passed']:
        raise RuntimeError('formal membership audit must pass first')
    lock = json.loads((art / 'CONFIG_LOCK.json').read_text())
    import csv
    stage_rows = {}
    for r in csv.DictReader((art / a.stages).open()):
        stage_rows[(r['task'], int(r['seed']), r['arm'])] = r
    strata = {'rescue': [], 'harm': []}
    for task in lock['tasks']:
        for seed in lock['formal_seeds']:
            d = {}
            for arm in ('vanilla', 'attn_matched'):
                d[arm] = json.loads((art / 'episodes/formal'/task/arm /
                                     f'episode_{seed:03d}_summary.json').read_text())
            v, m = d['vanilla'], d['attn_matched']
            if m['success'] and not v['success']:
                strata['rescue'].append((task, seed, 'vanilla'))
            elif v['success'] and not m['success']:
                strata['harm'].append((task, seed, 'attn_matched'))
    rng = np.random.default_rng(20241017)
    panels = []
    for name in ('rescue', 'harm'):
        items = strata[name]
        if not items:
            panels.append({'stratum': name, 'available': 0, 'selected': []})
            continue
        # the arm that FAILED determines the category we want to be representative of
        counts = {}
        for task, seed, failed_arm in items:
            row = stage_rows.get((task, seed, failed_arm))
            stage = row['stage'] if row else 'undetermined'
            counts[stage] = counts.get(stage, 0) + 1
        dominant = max(sorted(counts.items()), key=lambda kv: kv[1])[0]
        pool = [x for x in items if (stage_rows.get((x[0], x[1], x[2])) or {}).get('stage') == dominant]
        if len(pool) < 2:
            pool = items
        pick = [pool[i] for i in rng.choice(len(pool), size=min(2, len(pool)), replace=False)]
        selected = []
        for task, seed, failed_arm in pick:
            better = 'attn_matched' if failed_arm == 'vanilla' else 'vanilla'
            row = stage_rows.get((task, seed, failed_arm)) or {}
            selected.append({
                'task': task, 'seed': seed, 'failed_arm': failed_arm, 'better_arm': better,
                'failed_stage': row.get('stage'), 'failed_stage_basis': row.get('stage_basis'),
                'failed_termination': row.get('failure_reason'),
                'better_stage': (stage_rows.get((task, seed, better)) or {}).get('stage'),
                'videos': {arm: json.loads((art / 'episodes/formal'/task/arm /
                                            f'episode_{seed:03d}_summary.json').read_text())['video']
                           for arm in ('vanilla', 'attn_matched')},
            })
        panels.append({'stratum': name, 'available': len(items),
                       'stage_counts_of_failed_arm': counts, 'dominant_stage': dominant,
                       'selection_rule': 'representative of dominant failed stage; fixed RNG tie-break',
                       'selected': selected})
    out = {'note': ('Panels are illustrative examples chosen by a frozen rule; they are not '
                    'evidence of effect size and must not replace the paired statistics.'),
           'panels': panels}
    (art / 'QUALITATIVE_PANELS.json').write_text(json.dumps(out, indent=2) + '\n')
    for p in panels:
        print(p['stratum'], 'available', p['available'],
              [(s['task'], s['seed'], s['failed_stage']) for s in p['selected']])


if __name__ == '__main__':
    main()
