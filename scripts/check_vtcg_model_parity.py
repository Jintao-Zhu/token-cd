"""Same-state model parity on step 0: the CLEAN (positive) branch must produce the
same action scores for every arm of a case if the restored observation and the model
forward are identical. This is stronger than comparing PNG/state hashes, because it
covers preprocessing, proprio wiring and numerics.

Interpretation:
- exact match  -> the clean branch is reproducible across arms on this GPU;
- close match -> kernel/algorithm differences exist; report the actual deviation,
                 never round it to "identical";
- large diff  -> arm-specific input differences: investigate, do not average away.

A mismatch does NOT invalidate within-case pairing of the intervention arms, but it
must be reported. Never auto-repair or drop anything here.
"""
from pathlib import Path
import argparse, json
import numpy as np

ARMS = ['attn_matched', 'attn_fixed', 'attn_topp', 'kmeans_matched', 'kmeans_fixed',
        'kmeans_topp', 'vanilla', 'random_matched', 'norm_noise']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact', type=Path, required=True)
    ap.add_argument('--phase', choices=('engineering', 'formal'), default='engineering')
    ap.add_argument('--atol', type=float, default=0.0,
                    help='0.0 = require exact equality; >0 reports the deviation')
    a = ap.parse_args()
    root = a.artifact / 'episodes' / a.phase
    rows, pending, exact_cases = [], [], 0
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        seeds = sorted({int(p.name.split('_')[1]) for p in task_dir.glob('*/episode_*_arrays.npz')})
        for seed in seeds:
            base, ref_arm, missing = None, None, []
            for arm in ARMS:
                f = task_dir / arm / f'episode_{seed:03d}_arrays.npz'
                if not f.is_file():
                    missing.append(arm)
                    continue
                with np.load(f) as z:
                    first = np.asarray(z['positive_logits'][0])
                if base is None:
                    base, ref_arm = first, arm
                    continue
                delta = float(np.max(np.abs(first - base))) if first.shape == base.shape else float('inf')
                rows.append({'task': task_dir.name, 'seed': seed, 'arm': arm, 'reference_arm': ref_arm,
                             'shape_match': first.shape == base.shape, 'max_abs_delta_positive_logits_step0': delta})
            if missing:
                pending.append({'task': task_dir.name, 'seed': seed, 'missing_arms': missing})
            elif rows and all(r['max_abs_delta_positive_logits_step0'] == 0.0
                              for r in rows if r['task'] == task_dir.name and r['seed'] == seed):
                exact_cases += 1
    worst = max((r['max_abs_delta_positive_logits_step0'] for r in rows), default=None)
    failed = [r for r in rows if not r['shape_match'] or r['max_abs_delta_positive_logits_step0'] > a.atol] \
        if a.atol == 0.0 else []
    out = {'phase': a.phase, 'comparisons': len(rows), 'fully_exact_cases': exact_cases,
           'pending_cases': pending, 'exact_cases_with_all_nine_arms': exact_cases,
           'worst_max_abs_delta': worst, 'atol': a.atol,
           'shape_or_exactness_failures': failed,
           'note': ('Step-0 clean-branch scores only. Later steps diverge because each arm '
                    'perturbs its own trajectory after step 0; that divergence is expected.'),
           'detail': rows}
    (a.artifact / f'{a.phase.upper()}_MODEL_PARITY.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'detail'}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
