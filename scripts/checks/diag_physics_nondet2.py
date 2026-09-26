#!/usr/bin/env python3
"""Find the exact step where a full replay diverges."""
import os, json, numpy as np
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
from pathlib import Path
from libero.libero import benchmark, get_libero_path
from research.semantic_token_cd.libero_same_state_fork_worker import make_env
from research.semantic_token_cd.same_state_fork_common import array_sha256

suite = benchmark.get_benchmark_dict()["libero_90"]()
base = ("artifacts/libero90_five_task_simpler_config_v1/episodes/"
        "KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it/vanilla")
ep = json.load(open(f"{base}/init_000.json"))
tid = ep["task_id"]; task = suite.get_task(tid)
init = np.asarray(suite.get_task_init_states(tid)[ep["init_state_id"]])
bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
acts = np.asarray(ep["trajectory"], float)

def full_replay(env):
    env.reset(); env.set_init_state(init)
    for _ in range(10): env.step([0,0,0,0,0,0,-1])
    traj=[]
    for i in range(len(acts)):
        env.step(acts[i].tolist())
        traj.append(np.asarray(env.get_sim_state()).copy())
    return traj, bool(env.check_success())

env = make_env(bddl)
t1, s1 = full_replay(env)
t2, s2 = full_replay(env)
print(f"success: run1={s1} run2={s2}")
first=None
for i,(a,b) in enumerate(zip(t1,t2)):
    if not np.array_equal(a,b):
        first=i; break
print(f"first diverging step: {first}  (episode len {len(acts)})")
if first is not None:
    print(f"  max|diff| at that step = {np.abs(t1[first]-t2[first]).max():.3e}")
    print(f"  max|diff| at final step = {np.abs(t1[-1]-t2[-1]).max():.3e}")
    # which dof
    d=np.abs(t1[first]-t2[first])
    idx=np.argsort(d)[-5:]
    print(f"  top diverging dof indices: {idx.tolist()} vals={np.round(d[idx],6).tolist()}")
env.close()
