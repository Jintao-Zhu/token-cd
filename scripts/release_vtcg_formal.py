"""Engineering gate -> formal release. Refuses on any incomplete/failed evidence.
This script is the ONLY place that sets formal_status=approved_after_engineering.
It never fabricates an audit pass, and never edits CONFIG_LOCK while a queue runs.
"""
from pathlib import Path
import argparse, fcntl, hashlib, json, os, sys, time

TASKS = ['google_robot_open_drawer', 'google_robot_close_drawer',
         'google_robot_pick_coke_can', 'google_robot_move_near']


def fail(msg, reasons):
    reasons.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact', type=Path, required=True)
    ap.add_argument('--apply', action='store_true',
                    help='write ENGINEERING_AUDIT_PASS.json and update CONFIG_LOCK')
    a = ap.parse_args()
    art = a.artifact.resolve()
    lock_path = art / 'CONFIG_LOCK.json'
    lock_raw = lock_path.read_bytes()
    lock = json.loads(lock_raw)
    reasons = []
    arms = lock['arms']
    seeds = lock['engineering_seeds']
    expected = len(TASKS) * len(seeds) * len(arms)
    # 1. membership + technical + pairing
    seen, pair_hashes, missing = set(), {}, []
    for task in TASKS:
        for seed in seeds:
            sigs = set()
            for arm in arms:
                root = art / 'episodes' / 'engineering' / task / arm
                s = root / f'episode_{seed:03d}_summary.json'
                if not s.is_file():
                    missing.append(str(s)); continue
                d = json.loads(s.read_text())
                key = (task, seed, arm)
                seen.add(key)
                if not d.get('technical_pass'):
                    fail(f'technical_pass false: {key}', reasons)
                if d.get('formal_eligible'):
                    fail(f'engineering episode marked formal_eligible: {key}', reasons)
                for req, label in (('initial_state_sha256', 'state'), ('initial_rgb_sha256', 'rgb'),
                                   ('canonical_snapshot_sha256', 'canonical')):
                    if not d.get(req):
                        fail(f'missing {label} hash: {key}', reasons)
                sigs.add((d.get('initial_state_sha256'), d.get('initial_rgb_sha256'),
                          d.get('canonical_snapshot_sha256')))
                arrays = s.with_name(s.name.replace('_summary.json', '_arrays.npz'))
                if not arrays.is_file():
                    fail(f'missing arrays: {key}', reasons)
                elif hashlib.sha256(arrays.read_bytes()).hexdigest() != d.get('arrays_file_sha256'):
                    fail(f'arrays digest mismatch: {key}', reasons)
                video = d.get('video')
                if not video or not Path(video).is_file():
                    fail(f'missing video: {key}', reasons)
                else:
                    size = Path(video).stat().st_size
                    if size < 1000:
                        fail(f'suspiciously small video ({size}B): {key}', reasons)
            if len(sigs) != 1 and sigs:
                fail(f'pair hash mismatch {task}/{seed}: {sorted(sigs)}', reasons)
            if sigs:
                pair_hashes[f'{task}/{seed}'] = sorted(sigs)[0]
    if len(seen) != expected:
        fail(f'engineering membership {len(seen)}/{expected}', reasons)
    # 2. first-state parity
    fs = art / 'ENGINEERING_FIRST_STATE_AUDIT.json'
    if not fs.is_file():
        fail('missing ENGINEERING_FIRST_STATE_AUDIT.json', reasons)
    else:
        d = json.loads(fs.read_text())
        if not d.get('complete'):
            fail(f'first-state parity incomplete: {len(d.get("pending", []))} pending', reasons)
    # 3. media audit
    ma = art / 'ENGINEERING_MEDIA_AUDIT.json'
    if not ma.is_file():
        fail('missing ENGINEERING_MEDIA_AUDIT.json', reasons)
    else:
        d = json.loads(ma.read_text())
        errs = d.get('errors') or []
        if errs or d.get('error_count'):
            fail(f'media audit errors: {errs[:3]}', reasons)
        if 'episodes_checked' in d:  # ffprobe-based schema
            if d.get('episodes_checked') != d.get('expected_episodes'):
                fail(f'media audit incomplete {d.get("episodes_checked")}/{d.get("expected_episodes")}', reasons)
            if d.get('missing_count'):
                fail(f'media audit missing episodes {d.get("missing_count")}', reasons)
        elif not d.get('complete'):
            fail(f'media audit incomplete {d.get("observed")}/{d.get("expected")}', reasons)
    # 4. step-0 clean-branch model parity across all nine arms
    mp = art / 'ENGINEERING_MODEL_PARITY.json'
    cases = len(TASKS) * len(seeds)
    if not mp.is_file():
        fail('missing ENGINEERING_MODEL_PARITY.json', reasons)
    else:
        d = json.loads(mp.read_text())
        if d.get('shape_or_exactness_failures'):
            fail(f'model parity failures: {d["shape_or_exactness_failures"][:2]}', reasons)
        if d.get('pending_cases'):
            fail(f'model parity pending: {len(d["pending_cases"])} cases', reasons)
        if d.get('exact_cases_with_all_nine_arms') != cases:
            fail(f'exact model-parity cases {d.get("exact_cases_with_all_nine_arms")}/{cases}', reasons)
        if d.get('worst_max_abs_delta') not in (0.0, 0):
            fail(f'nondeterministic clean branch, worst delta {d.get("worst_max_abs_delta")}', reasons)
    # 5. no live queue
    status = json.loads((art / 'QUEUE_STATUS.json').read_text()) if (art / 'QUEUE_STATUS.json').is_file() else {}
    if status.get('status') == 'running' and status.get('phase') == 'engineering':
        pid = status.get('pid')
        if pid and Path(f'/proc/{pid}').exists():
            fail(f'engineering queue still running (pid {pid})', reasons)
    report = {'protocol_id': lock['protocol_id'], 'phase': 'engineering',
              'expected_episodes': expected, 'observed_episodes': len(seen),
              'missing': sorted(missing)[:20], 'reasons': reasons,
              'pair_hashes': pair_hashes, 'checked_at': time.time(),
              'limitation': ('Passing this gate certifies membership, technical audits, '
                             'shot pairing on one restored initial state, media integrity and '
                             'same-state budget/ranking parity. It does NOT prove full trajectory '
                             'equivalence, and does not certify that success rates differ.')}
    if not a.apply:
        print(json.dumps({k: v for k, v in report.items() if k not in ('pair_hashes',)}, indent=2))
        return 0 if not reasons else 1
    if reasons:
        print(json.dumps(report, indent=2))
        print('REFUSING to release formal: engineering gate not satisfied')
        return 1
    out = art / 'ENGINEERING_AUDIT_PASS.json'
    if out.exists():
        raise SystemExit(f'refuse to overwrite existing {out}')
    tmp = out.with_name(out.name + '.tmp')
    tmp.write_text(json.dumps(report, indent=2) + '\n')
    tmp.replace(out)
    # CONFIG_LOCK must be unchanged since we read it, then minimally updated.
    if lock_path.read_bytes() != lock_raw:
        raise SystemExit('CONFIG_LOCK changed during audit; re-run')
    lock['formal_status'] = 'approved_after_engineering'
    lock['formal_release'] = {'approved_at': time.time(),
                              'engineering_audit': str(out),
                              'engineering_audit_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
                              'scope': 'formal seeds 500-599 released for 4 tasks x 9 arms'}
    tmp = lock_path.with_name(lock_path.name + '.tmp')
    tmp.write_text(json.dumps(lock, indent=2) + '\n')
    tmp.replace(lock_path)
    print(json.dumps({'released': True, 'audit': str(out), 'formal_status': lock['formal_status']}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
