#!/usr/bin/env python3
"""Isolate the continuation drift source. No model needed for R1/R2.

R1  replay the STORED actions all the way   -> tests physics determinism
R2  re-render the same sim state repeatedly -> tests renderer determinism
R3  replay to step S then re-render twice   -> does render history leak?
"""
import os, json, numpy as np
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
from pathlib import Path
from libero.libero import benchmark, get_libero_path
from research.semantic_token_cd.libero_same_state_fork_worker import make_env, replay_to_step
from research.semantic_token_cd.same_state_fork_common import array_sha256

suite = benchmark.get_benchmark_dict()["libero_90"]()
base = ("artifacts/libero90_five_task_simpler_config_v1/episodes/"
        "KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it/vanilla")
ep = json.load(open(f"{base}/init_000.json"))
tid = ep["task_id"]; task = suite.get_task(tid)
init = suite.get_task_init_states(tid)[ep["init_state_id"]]
bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
acts = np.asarray(ep["trajectory"], float)
print(f"stored episode: steps={ep['steps']} success={ep['success']}")

print("\nR1) replay STORED actions from the init state, 3x  (physics determinism)")
env = make_env(bddl)
for k in range(3):
    env.reset(); obs = env.set_init_state(init)
    for _ in range(10): obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    for i in range(len(acts)):
        obs, _r, done, _ = env.step(acts[i].tolist())
        if done: break
    print(f"   run{k}: success={bool(env.check_success())} steps={i+1} "
          f"state_sha={array_sha256(np.asarray(env.get_sim_state()))[:10]}")

print("\nR2) same sim state, re-render repeatedly (renderer determinism)")
obs = replay_to_step(env, init, acts, 44)
state = np.asarray(env.get_sim_state()).copy()
first = None
for k in range(5):
    o = env.set_init_state(state)
    im = np.asarray(o["agentview_image"]).astype(int)
    if first is None: first = im
    d = np.abs(im - first)
    print(f"   {k}: sha={array_sha256(im)[:8]} maxdiff={d.max():4d} pix_diff={(d>0).sum():6d}")

print("\nR3) render the same state twice back-to-back (history leak?)")
for k in range(3):
    o1 = env.set_init_state(state); i1 = np.asarray(o1["agentview_image"]).astype(int)
    o2 = env.set_init_state(state); i2 = np.asarray(o2["agentview_image"]).astype(int)
    print(f"   pair{k}: maxdiff={np.abs(i1-i2).max():4d}")
env.close()
