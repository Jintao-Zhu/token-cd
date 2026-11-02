"""Simulator-event-derived failure-stage labels. Deterministic and reproducible.

These are NOT human annotations and NOT claims about latent model intent. Each label
is derived only from the simulator's own event stream saved in *_step_infos.json, so
it can be recomputed by anyone from the released artifacts. When the event stream
cannot support a stage, the episode is labelled `undetermined_*` instead of being
forced into a category. Absence of an event key is never treated as a negative event.

Stages (progress order, shallow -> deep):
 S1_approach, S2_engage, S3_manipulate, S4_final_state, S5_terminated_without_progress
 plus task-specific sub-labels and `undetermined_*` when signals are insufficient.
"""
from pathlib import Path
import argparse, csv, json

EPS = 0.005


def _series(info, key):
    return [row[key] for row in info if key in row]


def _first_true(values):
    for i, v in enumerate(values):
        if v is True:
            return i
    return None


def classify(task, info, failure_reason, success):
    if success:
        return 'success', 'episode reached the simulator success condition', {}
    n = len(info)
    ev = {}
    if task == 'google_robot_pick_coke_can':
        near = _series(info, 'near_obj')
        grasped = _series(info, 'is_grasped')
        lifted = _series(info, 'lifted_object')
        sig = _series(info, 'lifted_object_significantly')
        if not near and not grasped and not lifted:
            return 'undetermined_missing_events', 'no grasp/lift event keys present', ev
        ev = {'near_obj_first': _first_true(near), 'is_grasped_first': _first_true(grasped),
              'lifted_first': _first_true(lifted), 'lifted_sig_first': _first_true(sig),
              'steps': n, 'termination': failure_reason}
        if _first_true(near) is None:
            return 'S1_approach', 'never reached object proximity', ev
        if _first_true(grasped) is None:
            return 'S2_engage', 'near object but never grasped', ev
        if _first_true(lifted) is None:
            return 'S3_manipulate', 'grasped but object never lifted', ev
        if _first_true(sig) is None:
            return 'S3_manipulate_insufficient_lift', 'lifted but never significantly', ev
        if _first_true(grasped) is not None and grasped[-1] is False:
            return 'S4_final_state_object_dropped', 'significant lift then grasp lost', ev
        return 'S4_final_state', 'significant lift achieved but success not reached', ev
    if task in ('google_robot_open_drawer', 'google_robot_close_drawer'):
        near = _series(info, 'near_drawer')
        qpos = [float(v) for v in _series(info, 'qpos')]
        if not qpos:
            return 'undetermined_missing_events', 'no drawer joint position recorded', ev
        q0 = qpos[0]
        opening = task == 'google_robot_open_drawer'
        delta = [q - q0 for q in qpos]
        travel = (max(delta) if opening else min(delta))
        wrong = (min(delta) if opening else max(delta))
        ev = {'qpos_initial': q0, 'qpos_final': qpos[-1], 'qpos_min': min(qpos), 'qpos_max': max(qpos),
              'near_drawer_first': _first_true(near), 'steps': n, 'termination': failure_reason}
        if near and _first_true(near) is None:
            return 'S1_approach', 'never reached drawer proximity', ev
        if travel <= EPS and wrong >= -EPS:
            return 'S2_engage', 'drawer joint never moved beyond tolerance', ev
        if travel <= EPS:
            return 'S2_engage_motion_against_task', 'joint moved only opposite to the task direction', ev
        if wrong >= -EPS:
            return 'S3_manipulate', f'joint moved {travel:+.3f} in task direction but not to success', ev
        return 'S3_manipulate_with_counter_motion', f'travel {travel:+.3f}, opposite excursion {wrong:+.3f}', ev
    if task == 'google_robot_move_near':
        closest = _series(info, 'is_closest_to_tgt')
        correct = _series(info, 'moved_correct_obj')
        wrong = _series(info, 'moved_wrong_obj')
        near = _series(info, 'near_tgt_obj')
        height = _series(info, 'all_obj_keep_height')
        if not closest and not correct and not wrong:
            return 'undetermined_missing_events', 'no move-near event keys present', ev
        ev = {'is_closest_first': _first_true(closest), 'moved_correct_first': _first_true(correct),
              'moved_wrong_first': _first_true(wrong), 'near_tgt_first': _first_true(near),
              'all_obj_keep_height_all_true': all(height) if height else None,
              'steps': n, 'termination': failure_reason}
        if _first_true(closest) is None:
            return 'S1_approach', 'robot never became closest to the target object', ev
        if _first_true(correct) is None:
            return ('S2_wrong_object_interaction' if _first_true(wrong) is not None
                    else 'S2_engage'), 'closest but never moved the correct object', ev
        if _first_true(near) is None:
            return 'S3_manipulate', 'moved correct object but never brought it near the target', ev
        return 'S4_final_state', 'correct object near target but success condition unmet', ev
    return 'undetermined_unknown_task', f'no stage rules for {task}', ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact', type=Path, required=True)
    ap.add_argument('--phase', choices=('engineering', 'formal'), required=True)
    a = ap.parse_args()
    rows = []
    for f in sorted((a.artifact / 'episodes' / a.phase).glob('*/*/*_step_infos.json')):
        arm = f.parent.name
        task = f.parent.parent.name
        seed = int(f.name.split('_')[1])
        info = json.loads(f.read_text())
        s = f.with_name(f.name.replace('_step_infos.json', '_summary.json'))
        d = json.loads(s.read_text())
        stage, why, ev = classify(task, info, d.get('failure_reason'), bool(d.get('success')))
        rows.append({'task': task, 'seed': seed, 'arm': arm, 'success': bool(d.get('success')),
                     'stage': stage, 'stage_basis': why, 'failure_reason': d.get('failure_reason') or '',
                     'steps': len(info), **{f'ev_{k}': v for k, v in ev.items()}})
    if not rows:
        raise SystemExit('no step_infos found')
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    out = a.artifact / f'{a.phase.upper()}_FAILURE_STAGES.csv'
    with out.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    # paired stage-shift table (vanilla vs attn_matched), only where both exist
    order = ['S1_approach', 'S2_engage', 'S2_engage_motion_against_task', 'S2_wrong_object_interaction',
             'S2_engage' , 'S3_manipulate', 'S3_manipulate_insufficient_lift', 'S3_manipulate_with_counter_motion',
             'S4_final_state', 'S4_final_state_object_dropped', 'success']
    rank = {s: i for i, s in enumerate(dict.fromkeys(order))}
    pairs = {}
    for r in rows:
        pairs.setdefault((r['task'], r['seed']), {})[r['arm']] = r
    shift = []
    for (task, seed), arms in sorted(pairs.items()):
        v, m = arms.get('vanilla'), arms.get('attn_matched')
        if not v or not m:
            continue
        rv, rm = rank.get(v['stage']), rank.get(m['stage'])
        shift.append({'task': task, 'seed': seed, 'vanilla_stage': v['stage'], 'attn_matched_stage': m['stage'],
                      'stage_delta': (None if (rv is None or rm is None) else rm - rv),
                      'vanilla_success': v['success'], 'attn_matched_success': m['success']})
    (a.artifact / f'{a.phase.upper()}_STAGE_SHIFT.json').write_text(json.dumps(
        {'note': ('Stage order is a progress proxy defined by simulator events only; '
                  'deeper is not automatically better intent, and undetermined labels are excluded '
                  'from the delta. These labels are not human annotations.'),
         'summary': _stage_summary(rows, rank),
         'rank_order': list(rank), 'pairs': shift}, indent=2) + '\n')
    print(f'rows {len(rows)} pairs {len(shift)} -> {out}')


def _stage_summary(rows, rank):
    """Arm x stage contingency, per task and overall, plus paired stage-shift stats."""
    table = {}
    for r in rows:
        table.setdefault(r['task'], {}).setdefault(r['arm'], {}).setdefault(r['stage'], 0)
        table[r['task']][r['arm']][r['stage']] += 1
    overall = {}
    for r in rows:
        overall.setdefault(r['arm'], {}).setdefault(r['stage'], 0)
        overall[r['arm']][r['stage']] += 1
    pairs = {}
    for r in rows:
        pairs.setdefault((r['task'], r['seed']), {})[r['arm']] = r
    deltas = {}
    for (task, seed), arms in pairs.items():
        v, m = arms.get('vanilla'), arms.get('attn_matched')
        if not v or not m:
            continue
        deltas.setdefault(task, []).append((v['stage'], m['stage']))
    paired = {}
    for task, items in deltas.items():
        deeper = shallower = equal = undetermined = 0
        for v, m in items:
            rv, rm = rank.get(v), rank.get(m)
            if rv is None or rm is None:
                undetermined += 1
            elif rm > rv:
                deeper += 1
            elif rv > rm:
                shallower += 1
            else:
                equal += 1
        paired[task] = {'n_pairs': len(items), 'attn_matched_deeper': deeper,
                        'vanilla_deeper': shallower, 'same_stage': equal,
                        'undetermined': undetermined}
    return {'by_task_and_arm': table, 'overall_by_arm': overall,
            'paired_outcome_by_stage': paired,
            'note': ('Counts of event-derived stage labels; "deeper" uses the frozen stage rank '
                     'where success is the deepest state. Undetermined labels are kept visible, '
                     'not merged. These are not human annotations.')}


if __name__ == '__main__':
    main()
