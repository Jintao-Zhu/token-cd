#!/usr/bin/env python3
"""Validate and summarize the paired 300-seed RT-depth0 drawer study."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path('/home/leju-suzhou/zjt_ws/token-cd')
CANON = ROOT / 'artifacts/vanilla_recon_shr_canonical_0_299_v2'
OLD = ROOT / 'artifacts/l11_rank_causal_effect_study_v1_20260926'
TASKS = ('google_robot_open_drawer', 'google_robot_close_drawer')
ARMS = ('vanilla', 'l11_matched')
PROTOCOL = 'SIMPLER_RT_DEPTH0_DRAWERS_L11_MATCHED_PAIRED_0_299_V1'
RENDERER = 'SAPIEN RT depth0, spp=32, denoiser=true'


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def output_path(out: Path, task: str, arm: str, seed: int) -> Path:
    return out / 'episodes' / task / arm / f'episode_{seed:03d}_summary.json'


def validate_pair(out: Path, task: str, seed: int) -> dict:
    canon = read_json(CANON / 'episodes' / task / 'vanilla' / f'episode_{seed:03d}_summary.json')
    rows = {}
    for arm in ARMS:
        p = output_path(out, task, arm, seed)
        if not p.exists():
            raise RuntimeError(f'missing {p}')
        d = read_json(p)
        expected = {
            'protocol_id': PROTOCOL,
            'renderer': RENDERER,
            'task': task,
            'seed': seed,
            'arm': arm,
            'canonical_snapshot_sha256': canon['canonical_snapshot_sha256'],
            'initial_state_sha256': canon['initial_state_sha256'],
        }
        for k, v in expected.items():
            if d.get(k) != v:
                raise RuntimeError(f'{p}: {k}={d.get(k)!r}, expected {v!r}')
        rows[arm] = d
    if rows['vanilla'].get('initial_rgb_sha256') != rows['l11_matched'].get('initial_rgb_sha256'):
        raise RuntimeError(f'paired RT0 RGB mismatch: {task}:{seed}')
    # Compare against the previously audited RT0 drawer cohort wherever it exists.
    if 100 <= seed <= 129:
        old_v = read_json(OLD / 'closedloop_simpler' / task / 'vanilla' / f'seed_{seed:03d}.json')
        old_l = read_json(OLD / 'closedloop_simpler_l11matched_rt0' / task / 'l11_matched' / f'seed_{seed:03d}_summary.json')
        for arm, old in (('vanilla', old_v), ('l11_matched', old_l)):
            d = rows[arm]
            if (old.get('renderer') != RENDERER or
                old.get('canonical_snapshot_sha256') != d.get('canonical_snapshot_sha256') or
                old.get('initial_state_sha256') != d.get('initial_state_sha256') or
                old.get('initial_rgb_sha256_rt0') != d.get('initial_rgb_sha256')):
                raise RuntimeError(f'archived RT0 identity mismatch: {task}:{seed}:{arm}')
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--preflight-only', action='store_true')
    g.add_argument('--finalize', action='store_true')
    a = ap.parse_args()
    out = a.output.resolve()
    if a.preflight_only:
        cases = [('google_robot_open_drawer', 0), ('google_robot_close_drawer', 0), ('google_robot_close_drawer', 1)]
        for task, seed in cases:
            validate_pair(out, task, seed)
        payload = {'status': 'PASS', 'protocol_id': PROTOCOL, 'renderer': RENDERER,
                   'preflight_cases': [{'task': t, 'seed': s} for t, s in cases],
                   'verified_arms': list(ARMS)}
        (out / 'PRE_FLIGHT_PASS.json').write_text(json.dumps(payload, indent=2) + '\n')
        print(json.dumps(payload, indent=2))
        return

    result = {'protocol_id': PROTOCOL, 'complete': True, 'renderer': RENDERER,
              'seeds_per_task': 300, 'episodes': 1200, 'per_task': {}}
    for task in TASKS:
        successes = {arm: 0 for arm in ARMS}
        rescue = harm = 0
        for seed in range(300):
            pair = validate_pair(out, task, seed)
            for arm in ARMS:
                successes[arm] += bool(pair[arm]['success'])
            v, l = pair['vanilla']['success'], pair['l11_matched']['success']
            rescue += (not v and l)
            harm += (v and not l)
        result['per_task'][task] = {
            'n': 300,
            'vanilla_successes': successes['vanilla'],
            'vanilla_rate': successes['vanilla'] / 300,
            'l11_matched_successes': successes['l11_matched'],
            'l11_matched_rate': successes['l11_matched'] / 300,
            'rescue': rescue, 'harm': harm, 'paired_net': rescue - harm,
        }
    total_v = sum(x['vanilla_successes'] for x in result['per_task'].values())
    total_l = sum(x['l11_matched_successes'] for x in result['per_task'].values())
    total_r = sum(x['rescue'] for x in result['per_task'].values())
    total_h = sum(x['harm'] for x in result['per_task'].values())
    result['overall'] = {
        'n': 600,
        'vanilla_successes': total_v, 'vanilla_rate': total_v / 600,
        'l11_matched_successes': total_l, 'l11_matched_rate': total_l / 600,
        'rescue': total_r, 'harm': total_h, 'paired_net': total_r - total_h,
    }
    (out / 'FINAL_RESULTS.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    lines = [
        '# SIMPLER RT-depth0 Open/Close Drawer: 300 paired seeds', '',
        'Renderer: SAPIEN RT depth0, 32 spp, denoiser enabled. Seeds 0–299 per task; same historical Vanilla and VTCG protocols; paired canonical snapshots.', '',
        '| Task | Vanilla | VTCG | Rescue | Harm | Net |',
        '|---|---:|---:|---:|---:|---:|'
    ]
    for task, x in result['per_task'].items():
        label = task.replace('google_robot_', '').replace('_', ' ').title()
        lines.append(f"| {label} | {x['vanilla_successes']}/300 ({x['vanilla_rate']:.1%}) | {x['l11_matched_successes']}/300 ({x['l11_matched_rate']:.1%}) | {x['rescue']} | {x['harm']} | {x['paired_net']:+d} |")
    x = result['overall']
    lines += ['', f"**Both tasks:** Vanilla {x['vanilla_successes']}/600 ({x['vanilla_rate']:.1%}); VTCG {x['l11_matched_successes']}/600 ({x['l11_matched_rate']:.1%}); Rescue {x['rescue']}, Harm {x['harm']}, net {x['paired_net']:+d}.", '', 'All 1,200 episode summaries passed renderer, snapshot, state, paired-RGB, protocol, and historical-seed identity checks.']
    (out / 'FINAL_RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
