#!/usr/bin/env python3
"""Offline diagnosis of the completed LIBERO-90 SIMPLER-config matched run.

Reads existing episode traces/videos only.  No closed-loop experiment is run.
"""
from __future__ import annotations
import argparse, csv, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

GRID = 16
EPS = 1e-9


def load_json(path: Path):
    return json.loads(path.read_text())


def group_name(pair):
    v = bool(pair['vanilla_success']); m = bool(pair['matched_success'])
    if v and m: return 'both_success'
    if (not v) and m: return 'rescue'
    if v and (not m): return 'harm'
    return 'both_fail'


def action_l2(a, b, dims=6):
    aa = np.asarray(a[:dims], dtype=float); bb = np.asarray(b[:dims], dtype=float)
    return float(np.linalg.norm(aa - bb))


def action_inf(a, b, dims=6):
    aa = np.asarray(a[:dims], dtype=float); bb = np.asarray(b[:dims], dtype=float)
    return float(np.max(np.abs(aa - bb))) if aa.size else 0.0


def raw_gripper_env_value(raw):
    return -1.0 if float(raw[6]) > 0.5 else 1.0


def guided_change_fields(trace_item):
    raw = np.asarray(trace_item['raw_action'], dtype=float)
    executed = np.asarray(trace_item['executed_action'], dtype=float)
    trans = float(np.linalg.norm(executed[:3] - raw[:3]))
    rot = float(np.linalg.norm(executed[3:6] - raw[3:6]))
    raw_g = raw_gripper_env_value(raw)
    exec_g = float(executed[6])
    return trans, rot, bool(raw_g != exec_g), raw_g, exec_g


def mask_stats(tokens):
    tokens = sorted({int(x) for x in tokens})
    if not tokens:
        return {'m':0,'components':0,'bbox_r0':None,'bbox_r1':None,'bbox_c0':None,'bbox_c1':None,
                'row_mean':None,'col_mean':None,'top_fraction':None,'bottom_fraction':None,'spatial_entropy':None}
    cells = set(tokens)
    components = 0
    remaining = set(cells)
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            x = stack.pop(); r,c = divmod(x, GRID)
            for y in (x-GRID if r>0 else None, x+GRID if r<GRID-1 else None,
                      x-1 if c>0 else None, x+1 if c<GRID-1 else None):
                if y is not None and y in remaining:
                    remaining.remove(y); stack.append(y)
    rs = np.asarray([x//GRID for x in cells]); cs = np.asarray([x%GRID for x in cells])
    hist = np.zeros(GRID, dtype=float)
    counts = np.bincount(rs, minlength=GRID)
    hist = counts / max(1, counts.sum())
    entropy = float(-(hist[hist>0] * np.log(hist[hist>0])).sum())
    return {'m':len(cells),'components':components,'bbox_r0':int(rs.min()),'bbox_r1':int(rs.max()),
            'bbox_c0':int(cs.min()),'bbox_c1':int(cs.max()),'row_mean':float(rs.mean()),'col_mean':float(cs.mean()),
            'top_fraction':float(np.mean(rs < GRID/3)),'bottom_fraction':float(np.mean(rs >= 2*GRID/3)),
            'spatial_entropy':entropy}


def first_event(trace, predicate):
    for i,item in enumerate(trace):
        if predicate(i,item): return i
    return None


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--panels',type=int,default=12); a=ap.parse_args()
    root=a.root.resolve(); out=root; panels=out/'diagnostic_panels'; panels.mkdir(parents=True,exist_ok=True)
    pairs=[load_json(p) for p in sorted((root/'pairs').glob('*.json'))]
    all_step_rows=[]; event_rows=[]
    episode_summary={}

    for pair in pairs:
        group=group_name(pair)
        task=pair['task_name']; iid=int(pair['init_state_id'])
        vd=load_json(root/'episodes'/task/'vanilla'/f'init_{iid:03d}.json')
        md=load_json(root/'episodes'/task/'matched'/f'init_{iid:03d}.json')
        vt=vd.get('trace') or []; mt=md.get('trace') or []
        max_steps=int(md.get('max_policy_steps',400))
        # first executed-action divergence, and first sustained/major divergence
        first_div=None; first_major=None
        for i,(v,m) in enumerate(zip(vt,mt)):
            l2=action_l2(v['executed_action'],m['executed_action'])
            inf=action_inf(v['executed_action'],m['executed_action'])
            if first_div is None and (l2>1e-9 or abs(float(v['executed_action'][6])-float(m['executed_action'][6]))>1e-9):
                first_div=i
            if first_major is None and (l2>0.05 or abs(float(v['executed_action'][6])-float(m['executed_action'][6]))>0.5):
                first_major=i
            if first_div is not None and first_major is not None: break
        # first guidance action change on matched's own state
        # The released compact trace stores `guided_changed_dims`, while
        # `raw_action` is the final guided action before env postprocessing.
        # Exact clean-vs-guided action magnitudes require model replay and are
        # filled into the replay diagnostics rather than guessed here.
        guided=[i for i,item in enumerate(mt) if int(item.get('guided_changed_dims',0))>0]
        first_guided=guided[0] if guided else None
        # stepwise diagnostics for matched; controls included later in CSV
        prev_m=None
        for i,item in enumerate(mt):
            ms=mask_stats(item.get('selected_token_ids') or [])
            trans=rot=0.0; gflip=False; rawg=execg=None
            budget_jump=None if prev_m is None else int(ms['m']-prev_m)
            prev_m=ms['m']
            all_step_rows.append({
                'case_id':pair['case_id'],'group':group,'task':task,'init_state_id':iid,'step':i,
                'phase_fraction':i/max_steps,'m_effective':ms['m'],'m_raw':item.get('m_raw'),
                'budget_jump':budget_jump,'mask_components':ms['components'],
                'mask_bbox_r0':ms['bbox_r0'],'mask_bbox_r1':ms['bbox_r1'],'mask_bbox_c0':ms['bbox_c0'],'mask_bbox_c1':ms['bbox_c1'],
                'mask_row_mean':ms['row_mean'],'mask_col_mean':ms['col_mean'],'mask_top_fraction':ms['top_fraction'],
                'mask_bottom_fraction':ms['bottom_fraction'],'mask_spatial_entropy':ms['spatial_entropy'],
                'guided_translation':trans,'guided_rotation':rot,'guided_gripper_flip':int(gflip),
                'feature_perturbation_relative':item.get('feature_perturbation_relative'),
                'guided_changed_dims':item.get('guided_changed_dims'),
                'entities':json.dumps(item.get('entities'),ensure_ascii=False),
                'matched_cluster_ids':json.dumps(item.get('matched_cluster_ids')),
            })
        # events
        def at(i, field, default=None):
            return mt[i].get(field, default) if i is not None and i < len(mt) else default
        trans=rot=0.0; gflip=False
        event_rows.append({
            'case_id':pair['case_id'],'group':group,'task':task,'init_state_id':iid,
            'instruction':pair['instruction'],'vanilla_success':pair['vanilla_success'],'matched_success':pair['matched_success'],
            'vanilla_steps':pair['vanilla_steps'],'matched_steps':pair['matched_steps'],
            'first_executed_divergence_step':first_div,'first_major_divergence_step':first_major,
            'first_executed_divergence_time_s':None if first_div is None else first_div/30.0,
            'first_major_divergence_time_s':None if first_major is None else first_major/30.0,
            'first_guided_change_step':first_guided,'first_guided_change_time_s':None if first_guided is None else first_guided/30.0,
            'guided_translation_at_first':None,'guided_rotation_at_first':None,'guided_gripper_flip_at_first':None,
            'first_guided_phase_fraction':None if first_guided is None else first_guided/max_steps,
            'mean_m':float(np.mean([x.get('m_effective',0) for x in mt])) if mt else None,
            'max_budget_jump':max([abs(r['budget_jump']) for r in all_step_rows if r['case_id']==pair['case_id'] and r['budget_jump'] is not None], default=None),
            'mean_perturbation':float(np.mean([x.get('feature_perturbation_relative',0.0) for x in mt])) if mt else None,
            'max_perturbation':max([x.get('feature_perturbation_relative',0.0) for x in mt], default=None),
            'guided_change_fraction':float(np.mean([int(x.get('guided_changed_dims',0))>0 for x in mt])) if mt else None,
            'mean_guided_changed_dims':float(np.mean([x.get('guided_changed_dims',0) for x in mt])) if mt else None,
            'failed_arm':('matched' if group=='harm' else 'vanilla' if group=='rescue' else ''),
            'failed_arm_steps':pair['matched_steps'] if group=='harm' else pair['vanilla_steps'] if group=='rescue' else None,
            'failure_phase_heuristic':None,
            'observable_failure_type':'unclassified_requires_video_review',
            'confidence':'low',
            'evidence':f'first_div_step={first_div}; first_guided_step={first_guided}',
        })
        # episode summary
        episode_summary[(pair['case_id'], 'matched')]={
            'mean_m':event_rows[-1]['mean_m'],'max_budget_jump':event_rows[-1]['max_budget_jump'],
            'mean_perturbation':event_rows[-1]['mean_perturbation'],'max_perturbation':event_rows[-1]['max_perturbation'],
        }
    # failure phase heuristic based on failed arm step position
    for row in event_rows:
        if row['failed_arm_steps'] is not None:
            if row['first_major_divergence_step'] is not None:
                f=row['first_major_divergence_step']/400.0
                row['failure_phase_heuristic']='early' if f<0.3 else 'middle' if f<0.7 else 'late_or_timeout'
            else:
                row['failure_phase_heuristic']='no_major_divergence'
    # write CSVs
    event_fields=['case_id','group','task','init_state_id','instruction','vanilla_success','matched_success','vanilla_steps','matched_steps',
                  'first_executed_divergence_step','first_executed_divergence_time_s','first_major_divergence_step','first_major_divergence_time_s',
                  'first_guided_change_step','first_guided_change_time_s','guided_translation_at_first','guided_rotation_at_first','guided_gripper_flip_at_first',
                  'first_guided_phase_fraction','mean_m','max_budget_jump','mean_perturbation','max_perturbation','failed_arm','failed_arm_steps',
                  'guided_change_fraction','mean_guided_changed_dims','failure_phase_heuristic','observable_failure_type','confidence','evidence']
    with (out/'CASE_EVENT_TABLE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=event_fields); w.writeheader(); w.writerows(event_rows)
    state_fields=['case_id','group','task','init_state_id','step','phase_fraction','m_effective','m_raw','budget_jump','mask_components',
                  'mask_bbox_r0','mask_bbox_r1','mask_bbox_c0','mask_bbox_c1','mask_row_mean','mask_col_mean','mask_top_fraction','mask_bottom_fraction',
                  'mask_spatial_entropy','guided_translation','guided_rotation','guided_gripper_flip','feature_perturbation_relative','guided_changed_dims',
                  'entities','matched_cluster_ids']
    with (out/'STATE_DIAGNOSTICS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=state_fields); w.writeheader(); w.writerows(all_step_rows)
    # task/group summary
    summary=defaultdict(Counter)
    for row in event_rows:
        summary[(row['task'],row['group'])][row['failure_phase_heuristic'] or 'na']+=1
    with (out/'TASK_FAILURE_SUMMARY.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['task','group','n','early','middle','late_or_timeout','unclassified'])
        for (task,group),c in sorted(summary.items()):
            n=sum(c.values()); w.writerow([task,group,n,c['early'],c['middle'],c['late_or_timeout'],c['na']])
    # save compact summary for report
    preview={}
    for group in ('rescue','harm'):
        rows=[r for r in event_rows if r['group']==group]
        preview[group]={
            'n':len(rows),'first_guided_none':sum(r['first_guided_change_step'] is None for r in rows),
            'gripper_flip_any':sum(bool(r['guided_gripper_flip_at_first']) for r in rows),
            'guided_change_present':sum(r['first_guided_change_step'] is not None for r in rows),
            'mean_guided_changed_dims':float(np.mean([r['mean_guided_changed_dims'] or 0 for r in rows])) if rows else None,
            'phase_counts':Counter(r['failure_phase_heuristic'] for r in rows),
        }
    (out/'DIAGNOSTIC_QUICK.json').write_text(json.dumps(preview,indent=2,default=str)+'\n')
    print(json.dumps({'pairs':len(pairs),'event_rows':len(event_rows),'state_rows':len(all_step_rows),'preview':preview},indent=2,default=str))

if __name__=='__main__': main()
