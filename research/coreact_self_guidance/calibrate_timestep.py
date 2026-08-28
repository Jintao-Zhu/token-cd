from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare
from research.coreact_self_guidance.timestep_sampler import sample_timestep_shift_actions


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "rms": float(np.sqrt(np.mean(array**2))),
    }


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--workspace",type=Path,required=True); parser.add_argument("--artifact",type=Path,required=True); args=parser.parse_args(); workspace,artifact=args.workspace.resolve(),args.artifact.resolve()
    if (artifact/"phase0_calibration.json").exists(): raise FileExistsError("refusing to overwrite calibration")
    protocol=yaml.safe_load((artifact/"protocol.yaml").read_text()); shifts=[float(x) for x in protocol["shift_grid"]]
    config,policy,preprocessor,_=load_policy_and_processors(workspace); states=[json.loads(x) for x in (artifact/"phase0_state_manifest.jsonl").read_text().splitlines()]
    traces={shift:[] for shift in shifts}; state_hashes=[]
    for spec in states:
        env,env_preprocessor,_=make_task_env("libero_spatial",spec["task_id"],config)
        try:
            env.envs[0].init_state_id=spec["init_state_id"]; observation,_=env.reset(seed=spec["reset_seed"]); batch=prepare(policy,preprocessor,env_preprocessor,observation,spec["language"])
            tensors=[*batch["images"],*batch["image_masks"],batch["lang_tokens"],batch["lang_masks"],batch["state"]]; state_hashes.append({"state_id":spec["state_id"],"input_sha256":hashlib.sha256(b"".join(t.detach().cpu().contiguous().numpy().tobytes() for t in tensors)).hexdigest()})
            generator=torch.Generator(device=batch["state"].device).manual_seed(spec["action_noise_seed"]*1000); noise=torch.randn((1,config.chunk_size,config.max_action_dim),generator=generator,device=batch["state"].device,dtype=batch["state"].dtype)
            for shift in shifts:
                _,trace=sample_timestep_shift_actions(policy.model,batch["images"],batch["image_masks"],batch["lang_tokens"],batch["lang_masks"],batch["state"],noise,shift=shift,w=0.5)
                traces[shift].append(trace["step_traces"])
        finally: env.close()
    e1=workspace/"artifacts/coreact_ensemble_vs_contrast_control_v1_20260809_152414"; reference=[]
    for path in e1.glob("episodes/*.json"):
        record=json.loads(path.read_text())
        if record["arm"]=="B_toward_top8":
            for trace in record["replan_traces"]: reference.extend(trace["applied_delta_l2_norm_post_clip"])
    if not reference: raise RuntimeError("E1 reference missing")
    target=summarize(reference); candidates=[]
    for shift in shifts:
        flat=[step for trajectory in traces[shift] for step in trajectory]; pre=[x["clean_minus_shift_l2_norm_pre_clip"] for x in flat]; clipped=[x["clipped_direction_l2_norm"] for x in flat]; applied=[x["actual_applied_delta_l2_norm"] for x in flat]
        per_step=[]
        for step in range(10):
            rows=[trajectory[step] for trajectory in traces[shift]]
            per_step.append({"step":step,"tau":rows[0]["flow_time_tau"],"shifted_tau":rows[0]["shifted_noisier_tau"],"pre_clip":summarize([r["clean_minus_shift_l2_norm_pre_clip"] for r in rows]),"clipped_direction":summarize([r["clipped_direction_l2_norm"] for r in rows]),"actual_applied":summarize([r["actual_applied_delta_l2_norm"] for r in rows]),"active_fraction":float(np.mean([r["active_timestep_shift_bool"] for r in rows]))})
        trajectory_rms=[float(np.sqrt(np.mean(np.square([s["actual_applied_delta_l2_norm"] for s in trajectory])))) for trajectory in traces[shift]]; trajectory_mean=[float(np.mean([s["actual_applied_delta_l2_norm"] for s in trajectory])) for trajectory in traces[shift]]
        applied_summary=summarize(applied); candidates.append({"shift":shift,"pre_clip":summarize(pre),"clipped_direction":summarize(clipped),"actual_applied":applied_summary,"trajectory_rms":summarize(trajectory_rms),"trajectory_mean":summarize(trajectory_mean),"trajectory_integrated_l2_mean":float(np.mean([sum(s["actual_applied_delta_l2_norm"] for s in trajectory) for trajectory in traces[shift]])),"active_step_fraction":float(np.mean([x["active_timestep_shift_bool"] for x in flat])),"clipping_fraction":float(np.mean([x["trust_region_clipping_active_bool"] for x in flat])),"ratio_to_e1_trajectory_rms":applied_summary["rms"]/(target["rms"]+1e-12),"per_flow_step":per_step,"n_states":len(traces[shift]),"n_steps":len(flat)})
    legal=[x for x in candidates if x["active_step_fraction"]>=0.8 and 0.7<=x["ratio_to_e1_trajectory_rms"]<=1.4]; chosen=min(legal,key=lambda x:(abs(x["ratio_to_e1_trajectory_rms"]-1),x["shift"])) if legal else None
    result={"pass":chosen is not None,"reference":{"artifact":str(e1),"metric":"B_toward_top8 actual applied post-clip","summary":target,"n":len(reference)},"candidates":candidates,"selected":chosen,"state_hashes":state_hashes,"outcome_blind":True,"rollout_started":False}
    (artifact/"phase0_calibration.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    if chosen:
        protocol["locked_shift"]=chosen["shift"]; protocol["phase0_result"]={"trajectory_rms_ratio":chosen["ratio_to_e1_trajectory_rms"],"active_step_fraction":chosen["active_step_fraction"],"clipping_fraction":chosen["clipping_fraction"]}; (artifact/"protocol.lock.yaml").write_text(yaml.safe_dump(protocol,sort_keys=False),encoding="utf-8"); decision="PHASE0_CALIBRATED_READY_FOR_ROLLOUT_PROTOCOL"
    else: decision="CALIBRATION_GATE_FAILED_NO_ROLLOUT"
    (artifact/"decision.json").write_text(json.dumps({"decision":decision,"phase0_pass":chosen is not None,"rollout_started":False,"confirmation_claim_allowed":False},indent=2)+"\n")
    print(json.dumps({"decision":decision,"reference":target,"candidates":[{"shift":x["shift"],"trajectory_rms":x["actual_applied"]["rms"],"ratio":x["ratio_to_e1_trajectory_rms"],"active":x["active_step_fraction"]} for x in candidates],"selected_shift":chosen["shift"] if chosen else None},indent=2))


if __name__=="__main__":main()
