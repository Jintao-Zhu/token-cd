from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path

import yaml


TASK_IDS = (0, 1, 5, 6)
ARMS = ("A_vanilla", "N0_pure_negative", "W05_shrink", "W15_extrapolate", "W20_extrapolate", "REF_toward_top8")
W = {"A_vanilla":1.0,"N0_pure_negative":0.0,"W05_shrink":0.5,"W15_extrapolate":1.5,"W20_extrapolate":2.0,"REF_toward_top8":0.5}


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--workspace",type=Path,required=True); p.add_argument("--phase0-artifact",type=Path,required=True); p.add_argument("--base-artifact",type=Path,required=True); p.add_argument("--artifact",type=Path,required=True); args=p.parse_args(); workspace,phase0,base,artifact=args.workspace.resolve(),args.phase0_artifact.resolve(),args.base_artifact.resolve(),args.artifact.resolve()
    if artifact.exists(): raise FileExistsError(artifact)
    if json.loads((phase0/"decision.json").read_text())["decision"]!="PHASE0_CALIBRATED_READY_FOR_ROLLOUT_PROTOCOL": raise RuntimeError("phase0 did not pass")
    from libero.libero import benchmark
    suite=benchmark.get_benchmark_dict()["libero_spatial"](); task_manifest={str(task_id):{"task_id":task_id,"language":suite.get_task(task_id).language,"bddl_file":suite.get_task(task_id).bddl_file} for task_id in TASK_IDS}
    for d in ("episodes","corrections","logs","status"): (artifact/d).mkdir(parents=True)
    (artifact/"task_manifest.json").write_text(json.dumps(task_manifest,indent=2,sort_keys=True)+"\n")
    phase_protocol=yaml.safe_load((phase0/"protocol.lock.yaml").read_text()); base_protocol=yaml.safe_load((base/"protocol.lock.yaml").read_text()); shift=float(phase_protocol["locked_shift"])
    code_paths=[workspace/"research/coreact_closed_loop/guidance.py",workspace/"research/coreact_closed_loop/runtime.py",workspace/"research/coreact_self_guidance/sampler.py",workspace/"research/coreact_self_guidance/timestep_sampler.py",workspace/"research/coreact_self_guidance/prepare_timestep_confirmation.py",workspace/"research/coreact_self_guidance/integrity.py",workspace/"research/coreact_self_guidance/run.py",workspace/"research/coreact_self_guidance/analyze_timestep_confirmation.py"]
    checkpoint=workspace/"task1/.hf-cache/hub/models--lerobot--smolvla_libero/snapshots/31d453f7edd78c839a8bbc39744a292686daf0de"; mean_path=workspace/"artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt"
    protocol={"experiment_name":"coreact_timestep_self_guidance_new_task_confirmation_v1","created_at":datetime.now().astimezone().isoformat(),"stage":"method_heldout_task_replication_benchmark_tasks_may_be_historically_burned","confirmation_scope":"new to timestep-shift mechanism selection; not a globally pristine benchmark confirmation","tasks":[{"id":x,"selection":"fixed before reading timestep-shift outcomes; excludes development/calibration tasks 2,4,7,9"} for x in TASK_IDS],"suite":"libero_spatial","states_per_task":50,"arms":list(ARMS),"total_episodes":1200,"locked_shift":shift,"trust_region_kappa":0.25,"flow_steps":10,"chunk_size":50,"executed_actions_per_chunk":10,"maximum_control_steps":280,"w_values":W,"save_velocity_correction_vectors":True,"correction_vector_contract":{"raw":"v_clean-v_branch","applied":"actual signed correction added to v_clean","dtype":"float32","shape":"[replans,flow_steps,batch,chunk,real_action_dim]","real_action_dim":7,"storage":"one append-only torch sidecar per non-vanilla episode with SHA-256 in episode JSON"},"regime_rules":{"strong_contrastive":"N0-A CI upper<0 and max(W15-A,W20-A) CI lower>0; monotonic support if W20 point>=W15 point","directional_contrastive":"N0-A point<0 and max(W15-A,W20-A) point>0 but strong rule fails","strong_shrinkage":"N0-A CI contains 0 and W05-A CI lower>0 and W15/W20 do not have CI lower>0","directional_shrinkage":"N0-A CI contains 0 and W05-A point>max(W15-A,W20-A,0) but strong rule fails","null":"all N0/W05/W15/W20 CIs contain 0 and absolute point estimates<0.10","mixed":"none of the above"},"bootstrap_replicates":2000,"bootstrap_unit":"paired init state","paired_test":"exact McNemar","multiple_comparison":"Holm within task across N0/W05/W15/W20 versus vanilla","outcomes_must_not_change_tasks_or_parameters":True,"source_phase0":str(phase0),"source_phase0_sha256":sha256(phase0/"phase0_calibration.json"),"old_episode_results_reused":False,"checkpoint":{"repo":"lerobot/smolvla_libero","revision":"31d453f7edd78c839a8bbc39744a292686daf0de","config_sha256":sha256(checkpoint/"config.json"),"weights_sha256":sha256(checkpoint/"model.safetensors")},"calibration_mean_sha256":sha256(mean_path),"hashes":{"code":{str(path.relative_to(workspace)):sha256(path) for path in code_paths}}}
    (artifact/"protocol.lock.yaml").write_text(yaml.safe_dump(protocol,sort_keys=False),encoding="utf-8")
    manifest=[]
    for task_id in TASK_IDS:
        for init_state_id in range(50):
            seed=120_000_000+task_id*100_000+init_state_id*10; pair=f"task{task_id:02d}__init{init_state_id:02d}"
            for arm in ARMS: manifest.append({"episode_id":f"{pair}__{arm}","pair_id":pair,"suite":"libero_spatial","task_id":task_id,"init_state_id":init_state_id,"language":task_manifest[str(task_id)]["language"],"reset_seed":seed+1,"action_noise_seed":seed+2,"selection_seed":seed+3,"second_action_noise_seed":seed+4,"arm":arm,"w":W[arm],"shift":shift})
    with (artifact/"episode_manifest.jsonl").open("x") as f:
        for row in manifest: f.write(json.dumps(row,sort_keys=True)+"\n")
    (artifact/"environment.json").write_text(json.dumps({"created_at":protocol["created_at"],"hostname":platform.node(),"workspace_git_status":"unavailable: ownership/safe-directory restriction","phase0_artifact":str(phase0)},indent=2)+"\n")
    print(json.dumps({"artifact":str(artifact),"tasks":TASK_IDS,"episodes":len(manifest),"shift":shift},indent=2))


if __name__=="__main__":main()
