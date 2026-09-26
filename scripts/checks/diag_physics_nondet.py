#!/usr/bin/env python3
"""Locate the physics non-determinism: where does the replay diverge?"""
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

env = make_env(bddl)

print("T1) is the INIT STATE itself reproducible after reset()?")
for k in range(3):
    env.reset(); env.set_init_state(init)
    s = np.asarray(env.get_sim_state())
    print(f"   {k}: sha={array_sha256(s)[:12]} maxdiff_vs_init={np.abs(s-init).max():.3e}")

print("\nT2) after the 10 settle steps?")
for k in range(3):
    env.reset(); env.set_init_state(init)
    for _ in range(10): env.step([0,0,0,0,0,0,-1])
    s = np.asarray(env.get_sim_state())
    print(f"   {k}: sha={array_sha256(s)[:12]}")

print("\nT3) after N replay steps, at which N does divergence appear?")
ref = None
for N in (1, 5, 10, 20, 30, 44):
    shas = []
    for k in range(3):
        env.reset(); env.set_init_state(init)
        for _ in range(10): env.step([0,0,0,0,0,0,-1])
        for i in range(N): env.step(acts[i].tolist())
        s = np.asarray(env.get_sim_state())
        shas.append(array_sha256(s)[:10])
    uniq = len(set(shas))
    print(f"   N={N:<3} distinct_states={uniq}/3  {shas}")
env.close()
