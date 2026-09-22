#!/usr/bin/env python3
"""Deterministic replay diagnostics for the completed 500-episode run."""
from __future__ import annotations
import argparse, csv, json, os
from pathlib import Path
import numpy as np

TASK_OBJECTS={
 'KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it':('butter_2','wooden_cabinet_1'),
 'KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet':('akita_black_bowl_1','wooden_cabinet_1'),
 'LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket':('tomato_sauce_1','basket_1'),
 'LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate':('porcelain_mug_1','plate_1'),
 'STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy':('black_book_1','desk_caddy_1'),
}


def load(path: Path): return json.loads(path.read_text())


def body_pos(inner,name):
    return np.asarray(inner.sim.data.body_xpos[inner.obj_body_id[name]],dtype=float)


def grasped_objects(inner):
    out=[]
    for name,obj in inner.objects_dict.items():
        try:
            if inner._check_grasp(gripper=inner.robots[0].gripper, object_geoms=obj): out.append(name)
        except Exception:
            pass
    return out


def classify(summary, success):
    if success: return 'success'
    if summary['wrong_grasps']:
        return 'wrong_object_grasp'
    if summary['ever_target_grasp']:
        if summary['target_max_lift'] < 0.03:
            return 'target_not_lifted_after_grasp'
        if summary['grasp_lost_before_end']:
            return 'dropped_or_lost_target'
        return 'placement_or_subtask_failure'
    if summary['target_max_displacement'] < 0.01:
        return 'failed_to_approach_or_grasp_target'
    return 'pushed_or_moved_target_without_successful_grasp'


def replay_arm(root:Path, pair:dict, arm:str, env_cls, get_libero_path, apply_action=True):
    episode=load(root/'episodes'/pair['task_name']/arm/f"init_{pair['init_state_id']:03d}.json")
    from libero.libero import benchmark
    suite=benchmark.get_benchmark_dict()['libero_90'](); tid=int(pair['task_id']); task=suite.get_task(tid); init=suite.get_task_init_states(tid)[int(pair['init_state_id'])]
    bddl=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file)
    env=env_cls(bddl_file_name=bddl,camera_heights=256,camera_widths=256)
    try:
        env.seed(0); env.reset(); obs=env.set_init_state(init)
        for _ in range(10): obs,_,_,_=env.step([0,0,0,0,0,0,-1])
        inner=getattr(env,'env',env); target,dest=TASK_OBJECTS[pair['task_name']]
        t0=body_pos(inner,target); records=[]; success_step=None
        actions=episode.get('trajectory') or [x['executed_action'] for x in episode.get('trace') or []]
        for step,action in enumerate(actions):
            grasped=grasped_objects(inner); tp=body_pos(inner,target); dp=body_pos(inner,dest)
            success_now=bool(env.check_success())
            if success_now and success_step is None: success_step=step
            records.append({'step':step,'grasped':grasped,'target_pos':tp.tolist(),'target_z':float(tp[2]),
                            'target_disp':float(np.linalg.norm(tp-t0)),'dest_dist':float(np.linalg.norm(tp-dp)),
                            'success_now':success_now})
            if apply_action: obs,_,done,_=env.step(action)
            if done: break
        success=bool(env.check_success())
        target_grasps=[i for i,r in enumerate(records) if target in r['grasped']]
        wrong=[(i,x) for i,r in enumerate(records) for x in r['grasped'] if x!=target]
        initial_z=float(t0[2]); max_lift=max([r['target_z']-initial_z for r in records],default=0.0)
        final_grasp=bool(records and target in records[-1]['grasped'])
        summary={
            'arm':arm,'success':success,'steps':len(records),'success_step':success_step,
            'target_object':target,'destination_object':dest,
            'ever_target_grasp':bool(target_grasps),'first_target_grasp_step':target_grasps[0] if target_grasps else None,
            'target_grasp_steps':len(target_grasps),'target_max_lift':float(max_lift),
            'target_max_displacement':float(max([r['target_disp'] for r in records],default=0.0)),
            'final_target_grasp':final_grasp,'grasp_lost_before_end':bool(target_grasps and not final_grasp),
            'wrong_grasps':wrong,'ever_wrong_grasp':bool(wrong),
            'first_wrong_grasp_step':wrong[0][0] if wrong else None,
        }
        summary['failure_class']=classify(summary,success)
        return summary
    finally:
        env.close()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--all-controls',action='store_true'); a=ap.parse_args(); root=a.root.resolve()
    os.environ['MUJOCO_GL']='egl'; os.environ['PYOPENGL_PLATFORM']='egl'; os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    pairs=[load(p) for p in sorted((root/'pairs').glob('*.json'))]
    flips=[p for p in pairs if bool(p['vanilla_success'])!=bool(p['matched_success'])]
    controls=[]
    for task in sorted({p['task_name'] for p in pairs}):
        for grp,pred in [('both_success',lambda x:x['vanilla_success'] and x['matched_success']),('both_fail',lambda x:not x['vanilla_success'] and not x['matched_success'])]:
            controls.extend([p for p in pairs if p['task_name']==task and pred(p)][:5])
    cases=flips+controls
    rows=[]
    for idx,p in enumerate(cases):
        for arm in ('vanilla','matched'):
            print('replay',idx+1,'/',len(cases),p['case_id'],arm,flush=True)
            s=replay_arm(root,p,arm,OffScreenRenderEnv,get_libero_path)
            s.update({'case_id':p['case_id'],'task_id':p['task_id'],'task_name':p['task_name'],'init_state_id':p['init_state_id'],
                      'group':('rescue' if p['matched_success'] and not p['vanilla_success'] else 'harm' if p['vanilla_success'] and not p['matched_success'] else 'both_success' if p['vanilla_success'] and p['matched_success'] else 'both_fail')})
            rows.append(s)
    fields=['case_id','group','task_id','task_name','init_state_id','arm','success','steps','success_step','target_object','destination_object',
            'ever_target_grasp','first_target_grasp_step','target_grasp_steps','target_max_lift','target_max_displacement','final_target_grasp','grasp_lost_before_end',
            'ever_wrong_grasp','first_wrong_grasp_step','wrong_grasps','failure_class']
    with (root/'REPLAY_DIAGNOSTICS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:(json.dumps(r[k]) if k=='wrong_grasps' else r.get(k)) for k in fields})
    summary={}
    for group in ('rescue','harm','both_success','both_fail'):
        sub=[r for r in rows if r['group']==group]
        summary[group]={}
        for arm in ('vanilla','matched'):
            arr=[r for r in sub if r['arm']==arm]
            if not arr: continue
            summary[group][arm]={'n':len(arr),'success':sum(r['success'] for r in arr),
                'classes':{k:sum(r['failure_class']==k for r in arr) for k in sorted({r['failure_class'] for r in arr})}}
    (root/'REPLAY_SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
