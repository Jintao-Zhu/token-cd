#!/usr/bin/env python3
"""Where does the continuation drift come from?

Three tests on one state:
  A. same sim state -> render repeatedly IN THE SAME PROCESS. Identical?
  B. same sim state -> render in a FRESH ENV. Identical?
  C. does the policy, seeing the replayed image, reproduce the action the
     stored vanilla episode actually took at that step?
"""
import os, json, glob, numpy as np
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
from pathlib import Path
from libero.libero import benchmark, get_libero_path
from research.semantic_token_cd.libero_same_state_fork_worker import make_env, replay_to_step
from research.ar_token_counterfactual.libero_runtime import (
    load_policy, predict_action, prepare_agentview, prepare_env_action)
from research.semantic_token_cd.same_state_fork_common import array_sha256

def img_of(env, obs):
    return np.asarray(obs["agentview_image"]).copy()

suite = benchmark.get_benchmark_dict()["libero_90"]()
ep = json.load(open("artifacts/libero90_five_task_simpler_config_v1/episodes/"
                    "KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it/"
                    "vanilla/init_000.json"))
tid = ep["task_id"]; task = suite.get_task(tid)
init = suite.get_task_init_states(tid)[ep["init_state_id"]]
bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
acts = np.asarray(ep["trajectory"], float)
STEP = 44

env = make_env(bddl)
obs = replay_to_step(env, init, acts, STEP)
state = np.asarray(env.get_sim_state()).copy()
base = img_of(env, obs)
print(f"state sha = {array_sha256(state)[:16]}   (episode_len={ep['steps']})")

print("\nA) same sim state, re-render 5x IN SAME PROCESS")
for k in range(5):
    o = env.set_init_state(state)
    im = img_of(env, o)
    d = np.abs(im.astype(int) - base.astype(int))
    print(f"   {k}: sha={array_sha256(im)[:8]} maxdiff={d.max():4d} pixels_diff={(d>0).sum():6d}")

print("\nB) same sim state, re-render in a FRESH ENV (same process)")
for k in range(3):
    env2 = make_env(bddl)
    o = env2.set_init_state(state)
    im = img_of(env2, o)
    d = np.abs(im.astype(int) - base.astype(int))
    print(f"   {k}: sha={array_sha256(im)[:8]} maxdiff={d.max():4d} pixels_diff={(d>0).sum():6d}")
    env2.close()

print("\nC) does the policy reproduce the stored vanilla action at this step?")
ckpt = Path('/home/leju-suzhou/zjt_ws/checkpoints/libero/vq-vla-openvla-7b-libero90')
model, proc = load_policy(ckpt, Path('/home/leju-suzhou/zjt_ws/token-cd/third_party/openvla/prismatic/extern/hf'),
                          device='cuda:0', dataset_statistics_path=ckpt/'dataset_statistics.json',
                          unnorm_key='libero_90_no_noops')
stored = acts[STEP]                      # env action actually taken by the stored episode
print(f"   stored env action @step{STEP}: {np.round(stored,4).tolist()}")
for k in range(3):
    o = env.set_init_state(state)
    _, im224 = prepare_agentview(o)
    raw = predict_action(model, proc, im224, task.language, unnorm_key='libero_90_no_noops')
    envact = prepare_env_action(raw)
    print(f"   rederived  {k}: {np.round(envact,4).tolist()}  maxdiff_vs_stored={np.abs(envact-stored).max():.4f}")
env.close()
