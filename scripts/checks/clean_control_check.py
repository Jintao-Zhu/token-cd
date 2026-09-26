#!/usr/bin/env python3
"""Control: can the vanilla policy reproduce the stored vanilla outcome when it
takes over from a mid-trajectory fork state?

This is the correct null for the fork experiment.  Comparing branch success
against the *stored* vanilla outcome is only valid if the clean continuation
reproduces that outcome; otherwise the comparison conflates intervention effect
with continuation drift.
"""
import os, json, glob, numpy as np
os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("PYOPENGL_PLATFORM","egl")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD","1")
from pathlib import Path
from libero.libero import benchmark, get_libero_path
from research.semantic_token_cd.libero_same_state_fork_worker import make_env, replay_to_step
from research.ar_token_counterfactual.libero_runtime import (
    load_policy, predict_action, prepare_agentview, prepare_env_action)

PER_TASK = int(os.environ.get("PER_TASK", "3"))
suite = benchmark.get_benchmark_dict()["libero_90"]()
ckpt = Path('/home/leju-suzhou/zjt_ws/checkpoints/libero/vq-vla-openvla-7b-libero90')
model, proc = load_policy(ckpt, Path('/home/leju-suzhou/zjt_ws/token-cd/third_party/openvla/prismatic/extern/hf'),
                          device='cuda:0', dataset_statistics_path=ckpt/'dataset_statistics.json',
                          unnorm_key='libero_90_no_noops')

stats = {}
for task_dir in sorted(glob.glob('artifacts/libero90_five_task_simpler_config_v1/episodes/*/vanilla')):
    for f in sorted(glob.glob(task_dir + '/*.json'))[:PER_TASK]:
        ep = json.load(open(f))
        tid = ep['task_id']; task = suite.get_task(tid)
        init = suite.get_task_init_states(tid)[ep['init_state_id']]
        bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
        acts = np.asarray(ep['trajectory'], float)
        for frac in (0.25, 0.55, 0.85):
            step = int(round(frac * ep['steps']))
            if step >= ep['steps'] or step <= 0:
                continue
            env = make_env(bddl)
            obs = replay_to_step(env, init, acts, step)
            for _ in range(ep['steps'] - step):
                _, im = prepare_agentview(obs)
                raw = predict_action(model, proc, im, task.language, unnorm_key='libero_90_no_noops')
                obs, _r, done, _ = env.step(prepare_env_action(raw).tolist())
                if done:
                    break
            ok = bool(env.check_success()); env.close()
            s = stats.setdefault((tid, frac), {'n': 0, 'ok': 0, 'stored': 0})
            s['n'] += 1; s['ok'] += int(ok); s['stored'] += int(ep['success'])
            print(f"T{tid} frac={frac:.2f} init={ep['init_state_id']} stored={int(ep['success'])} "
                  f"clean_cont={int(ok)} running={s['ok']}/{s['n']}", flush=True)

print("\n=== clean continuation vs stored vanilla ===")
tot = {'n': 0, 'ok': 0, 'stored': 0}
for k in sorted(stats):
    s = stats[k]
    for key in tot: tot[key] += s[key]
    print(f"  T{k[0]:<4} frac={k[1]:.2f}  clean-continuation={s['ok']}/{s['n']}  stored-vanilla={s['stored']}/{s['n']}")
print(f"  TOTAL clean-continuation={tot['ok']}/{tot['n']}   stored-vanilla={tot['stored']}/{tot['n']}")
