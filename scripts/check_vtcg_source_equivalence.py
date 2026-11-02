"""Bitwise equivalence check: re-run episodes under a new shared-source revision vs the
archived episodes produced under the previous revision, using the SAME frozen snapshot.

Compares, per arm: restored initial state/RGB hashes, executed action matrix, per-step
clean-branch action logits, success flag and step count. Bitwise equality over a full
episode is strong evidence that the source change does not affect this code path.
Any difference is reported as a failure and must stop the experiment.
"""
from pathlib import Path
import argparse, json
import numpy as np

ARMS = ['vanilla', 'attn_matched', 'kmeans_topp']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', type=Path, required=True, help='artifact with archived episodes')
    ap.add_argument('--candidate', type=Path, required=True, help='artifact with re-run episodes')
    ap.add_argument('--task', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--arms', nargs='+', default=ARMS)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    report = {'task': a.task, 'seed': a.seed, 'arms': {}, 'failures': []}
    for arm in a.arms:
        row = {}
        rec = {'task': a.task, 'seed': a.seed, 'arm': arm}
        try:
            rs = json.loads((a.reference / 'episodes/formal'/a.task/arm/f'episode_{a.seed:03d}_summary.json').read_text())
            cs = json.loads((a.candidate / 'episodes/formal'/a.task/arm/f'episode_{a.seed:03d}_summary.json').read_text())
            for key in ('initial_state_sha256', 'initial_rgb_sha256', 'canonical_snapshot_sha256'):
                row[key] = rs[key] == cs[key]
                if not row[key]:
                    report['failures'].append(f'{arm}: {key} differs')
            row['reference_source_sha256'] = rs['source_sha256']
            row['candidate_source_sha256'] = cs['source_sha256']
            for key in ('success', 'steps'):
                row[key + '_equal'] = rs[key] == cs[key]
                row['reference_' + key] = rs[key]
                row['candidate_' + key] = cs[key]
                if rs[key] != cs[key]:
                    report['failures'].append(f'{arm}: {key} differs ({rs[key]} vs {cs[key]})')
            with np.load(a.reference / 'episodes/formal'/a.task/arm/f'episode_{a.seed:03d}_arrays.npz') as zr, \
                 np.load(a.candidate / 'episodes/formal'/a.task/arm/f'episode_{a.seed:03d}_arrays.npz') as zc:
                ar, ac = zr['executed_actions'], zc['executed_actions']
                row['actions_shape_equal'] = ar.shape == ac.shape
                row['actions_bitwise_equal'] = bool(ar.shape == ac.shape and np.array_equal(ar, ac))
                row['actions_max_abs_diff'] = float(np.max(np.abs(ar - ac))) if ar.shape == ac.shape else None
                pr, pc = zr['positive_logits'], zc['positive_logits']
                row['positive_logits_shape_equal'] = pr.shape == pc.shape
                row['positive_logits_bitwise_equal'] = bool(pr.shape == pc.shape and np.array_equal(pr, pc))
                row['positive_logits_max_abs_diff'] = float(np.max(np.abs(pr - pc))) if pr.shape == pc.shape else None
                if not row['actions_bitwise_equal']:
                    report['failures'].append(f'{arm}: executed actions differ (max {row["actions_max_abs_diff"]})')
                if not row['positive_logits_bitwise_equal']:
                    report['failures'].append(f'{arm}: clean-branch logits differ (max {row["positive_logits_max_abs_diff"]})')
        except FileNotFoundError as exc:
            report['failures'].append(f'{arm}: {exc}')
            row['error'] = str(exc)
        report['arms'][arm] = row
    report['passed'] = not report['failures']
    report['interpretation'] = (
        'Bitwise equality of restored hashes, all executed actions and all clean-branch logits across '
        'a complete episode means the shared-source change is behaviourally inert on this code path. '
        'A non-empty failures list forbids continuing the experiment under the new revision.')
    a.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'arms'}, indent=2))
    print(json.dumps(report['arms'], indent=2)[:2000])
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
