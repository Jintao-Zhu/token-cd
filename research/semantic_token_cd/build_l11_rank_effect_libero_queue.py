#!/usr/bin/env python3
"""Freeze unique, already-recorded OSMesa LIBERO vanilla traces for replay."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path('/home/leju-suzhou/zjt_ws/token-cd')
SOURCE_ROOTS = (
    ROOT / 'artifacts/libero_action_value_trace_pilot_v2_osmesa_40pair_20260925',
    ROOT / 'artifacts/libero_action_value_trace_pilot_v2_osmesa_ext40_init8_15_20260925',
    ROOT / 'artifacts/libero_action_value_gate_osmesa_pilot_v1_20260925',
)
TASK_IDS = {3, 10, 49, 72, 73}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    chosen = {}
    for root in SOURCE_ROOTS:
        for path in sorted(root.glob('episodes/*/vanilla/episode.json')):
            episode = json.loads(path.read_text())
            task_id = int(episode['task_id'])
            if task_id not in TASK_IDS:
                continue
            key = (task_id, int(episode['init_state_id']))
            if key not in chosen:
                chosen[key] = {
                    'case_id': episode['case_id'], 'task_id': task_id,
                    'task_name': episode['task_name'],
                    'init_state_id': int(episode['init_state_id']),
                    'episode_json': str(path.resolve()),
                }
    cases = [chosen[k] for k in sorted(chosen)]
    if args.smoke:
        selected = []
        seen = set()
        for case in cases:
            tid = case['task_id']
            if tid not in seen:
                selected.append(case)
                seen.add(tid)
        cases = selected
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(x, sort_keys=True) + '\n' for x in cases))
    print(json.dumps({
        'case_count': len(cases), 'unique_tasks': sorted({x['task_id'] for x in cases}),
        'per_task': {str(t): sum(x['task_id'] == t for x in cases) for t in sorted({x['task_id'] for x in cases})},
        'output': str(out),
    }, sort_keys=True))


if __name__ == '__main__':
    main()
