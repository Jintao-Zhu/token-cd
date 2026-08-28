#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from lerobot.envs.factory import make_env_pre_post_processors
from libero.libero.envs.utils import postprocess_model_xml
from research.coreact_closed_loop.runtime import env_config
from research.coreact_expert_direction.direction_audit import prepare_demo_state, rewrite_demo_xml
from research.coreact_region.segmented_runtime import make_segmented_env
from research.coreact_trained_weak.runtime import load_policy
from research.coreact_capacity_weak.metrics import exact_training_pair
from research.coreact_capacity_weak.run_offline import velocity


TIMES = tuple(round(1.0 - 0.1*i, 1) for i in range(10)); EPS=1e-12
SCOPES = {"full": (slice(0,50), (0,1,2,3,4,5,6)), "front10": (slice(0,10), (0,1,2,3,4,5,6)), "tail40": (slice(10,50), (0,1,2,3,4,5,6)), "full_translation": (slice(0,50),(0,1,2)), "full_rotation": (slice(0,50),(3,4,5)), "full_gripper": (slice(0,50),(6,)), "front10_translation": (slice(0,10),(0,1,2)), "front10_rotation": (slice(0,10),(3,4,5)), "front10_gripper": (slice(0,10),(6,))}


def decomposition(strong, weak, target, valid_steps, action_slice, dimensions):
    es=(strong-target)[action_slice][:,dimensions]; ew=(weak-target)[action_slice][:,dimensions]; valid=valid_steps[action_slice,None].expand_as(es)
    es,ew=es[valid].float(),ew[valid].float()
    if es.numel()==0: return None
    alpha=torch.dot(ew,es)/(torch.dot(es,es)+EPS); perpendicular=ew-alpha*es; ew_norm=torch.linalg.vector_norm(ew); es_norm=torch.linalg.vector_norm(es)
    return {"alpha":float(alpha),"alpha_gt_1":bool(alpha>1),"orthogonal_ratio":float(torch.linalg.vector_norm(perpendicular)/(ew_norm+EPS)),"error_cosine":float(torch.dot(ew,es)/(ew_norm*es_norm+EPS)),"weak_to_strong_error_norm":float(ew_norm/(es_norm+EPS)),"strong_error_norm":float(es_norm),"weak_error_norm":float(ew_norm)}


@torch.no_grad()
def evaluate(models,batch,state):
    actions=batch["actions"]; noises=torch.cat([torch.randn(actions.shape,generator=torch.Generator(device=actions.device).manual_seed(seed),device=actions.device,dtype=actions.dtype) for seed in state["noise_seeds"]]); actions3=actions.expand(3,-1,-1); valid=~batch["action_is_pad"][0]
    prefixes={name:model.embed_prefix(batch["images"],batch["image_masks"],batch["lang_tokens"],batch["lang_masks"],state=batch["state"]) for name,model in models.items()}; points=[]; position_values={candidate:{position:[] for position in range(50)} for candidate in models if candidate!="strong"}
    for flow_step,tau in enumerate(TIMES):
        timestep=torch.full((3,),tau,device=actions.device,dtype=actions.dtype); x_t,target=exact_training_pair(actions3,noises,timestep)
        predictions={name:velocity(model,prefix.expand(3,-1,-1),pad.expand(3,-1),att.expand(3,-1),x_t,timestep)[:,:,:7] for name,model in models.items() for prefix,pad,att in [prefixes[name]]}
        for candidate in (name for name in models if name!="strong"):
            for noise_ordinal in range(3):
                row={"candidate":candidate,"flow_step":flow_step,"timestep":tau,"noise_ordinal":noise_ordinal}
                for scope,(action_slice,dimensions) in SCOPES.items():
                    metrics=decomposition(predictions["strong"][noise_ordinal],predictions[candidate][noise_ordinal],target[noise_ordinal,:,:7],valid,action_slice,dimensions)
                    if metrics is not None:
                        for key,value in metrics.items(): row[f"{scope}_{key}"]=value
                points.append(row)
                for position in range(50):
                    if valid[position]: position_values[candidate][position].append(decomposition(predictions["strong"][noise_ordinal],predictions[candidate][noise_ordinal],target[noise_ordinal,:,:7],valid,slice(position,position+1),(0,1,2,3,4,5,6)))
    positions=[]
    for candidate,by_position in position_values.items():
        for position,values in by_position.items():
            if values: positions.append({"candidate":candidate,"position":position,"measurements":len(values),"mean_alpha":float(np.mean([x["alpha"] for x in values])),"median_alpha":float(np.median([x["alpha"] for x in values])),"p_alpha_gt_1":float(np.mean([x["alpha_gt_1"] for x in values])),"mean_orthogonal_ratio":float(np.mean([x["orthogonal_ratio"] for x in values])),"median_orthogonal_ratio":float(np.median([x["orthogonal_ratio"] for x in values])),"mean_error_cosine":float(np.mean([x["error_cosine"] for x in values]))})
    return points,positions,{"finite":all(np.isfinite(value) for row in points for value in row.values() if isinstance(value,float)),"point_rows":len(points),"position_rows":len(positions)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--workspace",type=Path,required=True);parser.add_argument("--artifact",type=Path,required=True);parser.add_argument("--task-ids",type=int,nargs="+",default=list(range(10)));parser.add_argument("--max-states-per-task",type=int);args=parser.parse_args();workspace,artifact=args.workspace.resolve(),args.artifact.resolve();states=[json.loads(line) for line in (artifact/"state_manifest.jsonl").read_text().splitlines() if json.loads(line)["task_id"] in args.task_ids]
    if args.max_states_per_task is not None: states=[row for task in args.task_ids for row in [x for x in states if x["task_id"]==task][:args.max_states_per_task]]
    capacity=workspace/"artifacts/coreact_capacity_weak_v1_20260815_202000"; source=workspace/"artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522"; paths={"strong":source/"training_run/trajectory/checkpoints/015000/pretrained_model","weak_a_8l":capacity/"training/weak_a_8l/checkpoints/015000/pretrained_model","weak_b_4l":capacity/"training/weak_b_4l/checkpoints/015000/pretrained_model"}
    os.chdir(workspace/"LIBERO");loaded={name:load_policy(path) for name,path in paths.items()};config,strong_policy,preprocessor,_=loaded["strong"];models={name:item[1].model for name,item in loaded.items()}
    for task_id in args.task_ids:
        task_states=[row for row in states if row["task_id"]==task_id]
        if not task_states:continue
        env_preprocessor,_=make_env_pre_post_processors(env_cfg=env_config("libero_spatial",task_id),policy_cfg=config);env=make_segmented_env("libero_spatial",task_id)
        try:
            with h5py.File(task_states[0]["demo_path"],"r") as handle:
                for index,state in enumerate(task_states,1):
                    output=artifact/"states"/f"{state['state_id']}.json"
                    if output.exists():continue
                    episode=handle["data"][state["demo_id"]];states_np,actions_np=np.asarray(episode["states"]),np.asarray(episode["actions"]);xml=episode.attrs["model_file"];xml=xml.decode() if isinstance(xml,bytes) else xml;env._env.reset();env._env.reset_from_xml_string(postprocess_model_xml(rewrite_demo_xml(xml,workspace),{},demo_generation=False));env._env.env.sim.reset();raw=env._env.regenerate_obs_from_state(states_np[state["resolved_frame"]]);batch=prepare_demo_state(strong_policy,preprocessor,env_preprocessor,env,raw,state["language"],actions_np,state["resolved_frame"]);points,positions,integrity=evaluate(models,batch,state);temporary=output.with_suffix(".tmp");temporary.write_text(json.dumps({"state":state,"points":points,"positions":positions,"integrity":integrity},sort_keys=True)+"\n");temporary.replace(output);print(json.dumps({"task":task_id,"state":index,"of":len(task_states),"points":len(points)}),flush=True)
        finally:env.close()


if __name__=="__main__":main()
