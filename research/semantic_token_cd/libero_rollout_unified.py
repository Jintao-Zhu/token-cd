#!/usr/bin/env python3
"""Unified, artifact-first LIBERO rollout engine.

The ``--prepare`` phase is model-free and creates canonical simulator state
and observations.  Arm runs consume those files, so baselines can be rerun or
new arms added without resetting the pairing protocol.
"""
from __future__ import annotations
import argparse, hashlib, json, pickle, time
from pathlib import Path
import numpy as np

ARMS=("vanilla","recon","shr")
def sha(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def task_obj(benchmark, name):
    from libero.libero import benchmark as bm
    key="libero_"+benchmark; suite=bm.get_benchmark_dict()[key]()
    return next(t for i in range(suite.n_tasks) if (t:=suite.get_task(i)).name==name)

def env_for(task):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl=Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file
    return OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)

def prepare_episode(env, root, benchmark, task, seed):
    d=root/benchmark/task/f"seed_{seed:06d}"; d.mkdir(parents=True,exist_ok=True)
    state_path=d/"initial_state.pkl"
    env.seed(seed); obs=env.reset(); state=env.get_sim_state()
    front=np.asarray(obs["agentview_image"]); wrist=np.asarray(obs.get("robot0_eye_in_hand_image", obs.get("eye_in_hand_image", front)))
    if not state_path.exists(): state_path.write_bytes(pickle.dumps(state, protocol=4))
    (d/"initial_obs").mkdir(exist_ok=True)
    from PIL import Image
    Image.fromarray(front).save(d/"initial_obs/front.png"); Image.fromarray(wrist).save(d/"initial_obs/wrist.png")
    meta={"benchmark":benchmark,"task":task,"seed":seed,"initial_state_hash":sha(np.asarray(state)),"rgb_hash":{"agentview":sha(front),"eye_in_hand":sha(wrist)},"instruction":task_obj(benchmark,task).language}
    (d/"hashes.json").write_text(json.dumps(meta["rgb_hash"],indent=2)+"\n"); (d/"instruction.txt").write_text(meta["instruction"]+"\n"); (d/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")

def run_arm(env, root, benchmark, task, seed, arm, checkpoint=None):
    d=root/benchmark/task/f"seed_{seed:06d}"; meta=json.loads((d/"metadata.json").read_text()); state=pickle.loads((d/"initial_state.pkl").read_bytes())
    env.reset(); obs=env.set_init_state(state)
    if sha(np.asarray(env.get_sim_state())) != meta["initial_state_hash"]: raise RuntimeError("canonical state hash mismatch")
    if arm!="vanilla": raise NotImplementedError("Recon/SHR policy factory is intentionally explicit; wire the locked SIMPLER implementations here before rollout")
    if checkpoint is None: raise ValueError("--checkpoint is required for arm vanilla")
    from research.ar_token_counterfactual.libero_runtime import load_policy,predict_action,prepare_agentview,prepare_env_action
    model,processor=load_policy(Path(checkpoint),Path(checkpoint).parents[2]); traj=[]; t0=time.perf_counter(); done=False
    for step in range(300):
        _,im=prepare_agentview(obs); action=prepare_env_action(predict_action(model,processor,im,meta["instruction"])); traj.append(action.copy()); obs,_,done,_=env.step(action.tolist())
        if done: break
    result={"benchmark":benchmark,"task":task,"seed":seed,"arm":arm,"success":bool(env.check_success()),"episode_length":len(traj),"inference_seconds":time.perf_counter()-t0,"initial_state_hash":meta["initial_state_hash"],"rgb_hash":meta["rgb_hash"]}
    np.save(d/"trajectory.npy",np.asarray(traj)); (d/f"{arm}_result.json").write_text(json.dumps(result,indent=2)+"\n")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--benchmark",choices=("spatial","object"),required=True); p.add_argument("--task",required=True); p.add_argument("--seed-manifest",type=Path,required=True); p.add_argument("--root",type=Path,default=Path("artifacts/libero")); p.add_argument("--arm",choices=ARMS); p.add_argument("--prepare",action="store_true"); p.add_argument("--checkpoint"); p.add_argument("--gpu",type=int,default=0); a=p.parse_args()
    import os; os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
    task=task_obj(a.benchmark,a.task); env=env_for(task); seeds=[int(x["seed"] if isinstance(x,dict) else x) for x in json.loads(a.seed_manifest.read_text())]
    for seed in seeds:
        prepare_episode(env,a.root,a.benchmark,a.task,seed)
        if a.arm: run_arm(env,a.root,a.benchmark,a.task,seed,a.arm,a.checkpoint)
    env.close()
if __name__=="__main__": main()
