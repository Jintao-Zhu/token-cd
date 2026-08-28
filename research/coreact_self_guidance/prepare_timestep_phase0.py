from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from datetime import datetime
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--workspace",type=Path,required=True); parser.add_argument("--source-artifact",type=Path,required=True); parser.add_argument("--artifact",type=Path,required=True); args=parser.parse_args()
    workspace,source,artifact=args.workspace.resolve(),args.source_artifact.resolve(),args.artifact.resolve()
    if artifact.exists(): raise FileExistsError(artifact)
    for directory in ("logs","status"): (artifact/directory).mkdir(parents=True)
    for name in ("phase0_state_manifest.jsonl","task_manifest.json"): shutil.copy2(source/name,artifact/name)
    code_paths=[workspace/"research/coreact_closed_loop/guidance.py",workspace/"research/coreact_closed_loop/runtime.py",workspace/"research/coreact_self_guidance/sampler.py",workspace/"research/coreact_self_guidance/timestep_sampler.py",workspace/"research/coreact_self_guidance/prepare_timestep_phase0.py",workspace/"research/coreact_self_guidance/calibrate_timestep.py"]
    protocol={"experiment_name":"coreact_timestep_self_guidance_phase0_v1","created_at":datetime.now().astimezone().isoformat(),"stage":"phase0_only_no_rollout","primary_construction":"fixed_x_change_timestep_only","time_convention":"SmolVLA tau=1 is noisier and tau=0 is cleaner","formula":{"clean":"v(x_tau,tau)","shift":"v(x_tau,min(1,tau+shift))"},"shift_grid":[0.02,0.05,0.1,0.2,0.3],"phase0_states":20,"phase0_tasks":[2,9],"flow_steps":10,"w_for_comparable_applied_metric":0.5,"trust_region_kappa":0.25,"real_action_dimensions":7,"metrics":["pre_clip_l2","clipped_direction_l2","actual_applied_l2","active_step_fraction","per_step_strength","trajectory_mean","trajectory_rms","trajectory_integrated_l2","clipping_fraction"],"calibration_target":"E1 B_toward_top8 actual applied post-clip trajectory perturbation","legal_ratio":[0.7,1.4],"outcome_blind":True,"rollout_forbidden_in_this_artifact":True,"source_artifact":str(source),"source_protocol_sha256":sha256(source/"protocol.lock.yaml"),"hashes":{"code":{str(p.relative_to(workspace)):sha256(p) for p in code_paths}}}
    (artifact/"protocol.yaml").write_text(yaml.safe_dump(protocol,sort_keys=False),encoding="utf-8")
    (artifact/"environment.json").write_text(json.dumps({"created_at":protocol["created_at"],"hostname":platform.node(),"source_artifact":str(source)},indent=2)+"\n")
    print(json.dumps({"artifact":str(artifact),"shift_grid":protocol["shift_grid"],"phase0_states":20},indent=2))


if __name__=="__main__":main()
