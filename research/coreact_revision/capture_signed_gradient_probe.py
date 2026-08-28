#!/usr/bin/env python3
"""Capture an expensive, target-dependent signed gradient diagnostic."""
from __future__ import annotations

import argparse, json
from collections import defaultdict
from pathlib import Path
import h5py, numpy as np, torch
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from research.coreact_closed_loop.runtime import env_config, load_policy_and_processors
from research.coreact_exploration.instrumentation import predict_teacher_forced_velocity
from research.coreact_exploration.run_development_gates import CAMERA_IDS, replacements_for
from research.coreact_region.segmented_runtime import batched_observation, make_segmented_env
from research.coreact_revision.run_region_sign_capture_v3 import action_chunk, rewrite_demo_xml


def read(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def append(path, row):
    with Path(path).open("a") as f: f.write(json.dumps(row, sort_keys=True) + "\n")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--workspace",type=Path,required=True); p.add_argument("--source-artifact",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--means",type=Path,required=True); a=p.parse_args()
    workspace, source, output = a.workspace.resolve(), a.source_artifact.resolve(), a.output.resolve(); output.parent.mkdir(parents=True,exist_ok=True)
    manifest={(r["task_id"],r["demo_id"],r["frame_id"]):r for r in read(source/"state_manifest.jsonl")}
    if len(manifest)!=300: raise RuntimeError(f"expected 300 states, got {len(manifest)}")
    task_names={r["task_id"]:r["name"] for r in json.loads((source/"task_manifest.json").read_text())}
    done={(r["task_id"],r["demo_id"],r["frame_id"],r["group_id"]) for r in read(output)} if output.exists() else set()
    config,policy,preprocessor,_=load_policy_and_processors(workspace); means=torch.load(a.means,weights_only=True,map_location="cpu"); dataset=workspace/"LIBERO/libero/datasets/libero_spatial"
    for task in sorted(task_names):
        env=make_segmented_env("libero_spatial",task); env_pre,_=make_env_pre_post_processors(env_cfg=env_config("libero_spatial",task),policy_cfg=config); task_rows=[r for k,r in manifest.items() if k[0]==task]
        try:
            with h5py.File(dataset/f"{task_names[task]}_demo.hdf5","r") as h5:
                groups=defaultdict(list)
                for r in task_rows: groups[r["demo_id"]].append(r)
                for demo in sorted(groups,key=lambda x:int(x.split("_")[-1])):
                    ep=h5["data"][demo]; states=np.asarray(ep["states"]); actions=np.asarray(ep["actions"]); xml=ep.attrs["model_file"]; xml=xml.decode() if isinstance(xml,bytes) else xml
                    env._env.reset(); env._env.reset_from_xml_string(__import__("libero.libero.envs.utils",fromlist=["postprocess_model_xml"]).postprocess_model_xml(rewrite_demo_xml(xml,workspace),{},demo_generation=False)); env._env.env.sim.reset()
                    for row in sorted(groups[demo],key=lambda x:x["frame_id"]):
                        frame=row["frame_id"]; selected=row["selected_groups"]
                        if all((task,demo,frame,f"prefix-{i}") in done for i in selected): continue
                        raw=env._env.regenerate_obs_from_state(states[frame]); obs=env._format_raw_obs(raw); act,pad=action_chunk(actions,frame)
                        batch=preprocess_observation(batched_observation(obs)); batch["task"]=[row["language"]]; batch["action"]=act.unsqueeze(0); batch["action_is_pad"]=pad.unsqueeze(0); batch=preprocessor(env_pre(batch)
                        )
                        images,masks=policy.prepare_images(batch); state=policy.prepare_state(batch); actions_t=policy.prepare_action(batch); gen=torch.Generator(device=actions_t.device).manual_seed(1729); noise=torch.randn(actions_t.shape,generator=gen,device=actions_t.device,dtype=actions_t.dtype); tau=torch.tensor([.5],device=actions_t.device,dtype=actions_t.dtype); holder={}
                        def capture(prefix,maps):
                            value=prefix.detach().requires_grad_(True); holder["prefix"]=value; holder["maps"]=maps[0]; return value
                        with torch.enable_grad():
                            base=predict_teacher_forced_velocity(policy.model,images,masks,batch["observation.language.tokens"],batch["observation.language.attention_mask"],state,actions_t,tau,noise,prefix_intervention=capture,record_attention=False,camera_ids=CAMERA_IDS)
                            valid=(~batch["action_is_pad"]).unsqueeze(-1).expand_as(base["u_target"]); diff=(base["v_pred"]-base["u_target"])**2; loss=(diff[valid]).mean(); grad=torch.autograd.grad(loss,holder["prefix"],retain_graph=False)[0][0].detach().cpu(); prefix=holder["prefix"].detach().cpu()[0]; span=holder["maps"]
                        repl=replacements_for(span,means,mode="position")
                        for index in selected:
                            token=span[index]; delta=(repl[index].detach().cpu()-prefix[index]); signed_dot=float((grad[index]*delta).sum()); append(output,{"task_id":task,"demo_id":demo,"frame_id":frame,"group_id":f"prefix-{index}","gradient_l2":float(grad[index].norm()),"gradient_dot_replacement_delta":signed_dot,"gradient_abs_dot":abs(signed_dot),"flow_condition":"tau0.5_noise1729"})
                        print(f"task={task} {demo} frame={frame} complete",flush=True)
        finally: env.close()
    print(f"captured {len(read(output))} gradient rows",flush=True)
if __name__=="__main__": main()
