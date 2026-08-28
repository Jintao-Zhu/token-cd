from __future__ import annotations

import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import torch

from research.coreact_closed_loop.run_pilot import vector_info_value
from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare
from research.coreact_self_guidance.reference_snapshot_gate import digest, fingerprints
from research.coreact_signed_utility.token_regions import sample_signed_region_actions

ARMS=("V","IT","IA","AT","AA","RT","RA")

def sha(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def atomic(path:Path,value)->None:
 t=path.with_suffix(path.suffix+".tmp"); t.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); t.replace(path)

def main()->None:
 p=argparse.ArgumentParser(); p.add_argument("--workspace",type=Path,required=True); p.add_argument("--artifact",type=Path,required=True); p.add_argument("--reference",type=Path,required=True); p.add_argument("--means",type=Path,required=True); p.add_argument("--unit-start",type=int,default=0); p.add_argument("--unit-stride",type=int,default=1); p.add_argument("--max-units",type=int); p.add_argument("--dry-run",action="store_true"); a=p.parse_args()
 ws,art,ref=a.workspace.resolve(),a.artifact.resolve(),a.reference.resolve(); decision=json.loads((art/"decision.json").read_text())["decision"]
 if decision not in ("TOKEN_DIRECTION_ORACLE_U0_CANDIDATES_LOCKED","TOKEN_DIRECTION_ORACLE_U0_DRYRUN_PASS"): raise RuntimeError(decision)
 os.environ.setdefault("HF_HOME",str(ws/"task1/.hf-cache")); os.environ["MUJOCO_GL"]="egl"
 rows=[json.loads(x) for x in (art/"episode_manifest.jsonl").read_text().splitlines()]; units={}
 for row in rows: units.setdefault(row["unit_id"],{})[row["arm"]]=row
 ordered=sorted(units.items())[a.unit_start::a.unit_stride]
 if a.max_units is not None: ordered=ordered[:a.max_units]
 if any(set(v)!=set(ARMS) for _,v in ordered): raise RuntimeError("arm manifest mismatch")
 cfg,policy,pre,post=load_policy_and_processors(ws); means=torch.load(a.means,weights_only=False,map_location="cpu"); visual=means["visual_position_mean"].to(cfg.device,dtype=torch.float32); cameras=tuple(means["camera_ids"]); protocol_sha=sha(art/"protocol.lock.yaml"); manifest_sha=sha(art/"episode_manifest.jsonl")
 complete=0
 for unit_id,specs in ordered:
  outputs=[art/"episodes"/f"{specs[x]['episode_id']}.json" for x in ARMS]
  if all(x.exists() for x in outputs): complete+=1; continue
  if any(x.exists() for x in outputs): raise RuntimeError(f"partial unit {unit_id}")
  sid=specs["V"]["snapshot_id"]; snap=torch.load(ref/"snapshots"/f"{sid}.pt",weights_only=False,map_location="cpu"); meta=snap["metadata"]; prefix=snap["action_prefix"].numpy(); records={}; branch={}; invalid=None
  for arm in ARMS:
   spec=specs[arm]; env,env_pre,env_post=make_task_env("libero_spatial",spec["task_id"],cfg); queue=[]; replans=0; actions=[]; success=False; reason="horizon"; intervention=[]; all_noise=[]; initial_clean=None
   try:
    inner=env.envs[0]; inner.init_state_id=spec["init_state_id"]; obs,_=env.reset(seed=int(meta["reset_seed"]));
    for action in prefix:
     obs,_,terminated,_,_=env.step(np.asarray(action,dtype=np.float32)[None,:])
     if bool(terminated[0]): raise RuntimeError("prefix terminated")
    batch=prepare(policy,pre,env_pre,obs,meta["instruction"]); fp=fingerprints(env,obs,batch,list(prefix)); g=torch.Generator(device=batch["state"].device).manual_seed(int(spec["noise_seed"])); first_noise=torch.randn((1,cfg.chunk_size,cfg.max_action_dim),generator=g,device=batch["state"].device,dtype=batch["state"].dtype)
    with torch.inference_mode(): clean=policy.model.sample_actions(batch["images"],batch["image_masks"],batch["lang_tokens"],batch["lang_masks"],batch["state"],noise=first_noise)
    initial_clean=digest(clean); branch[arm]={"fingerprints":fp,"reference_equal":fp==meta["reference_fingerprints"],"noise_sha256":digest(first_noise),"clean_chunk_sha256":initial_clean}
    remaining=max(0,280-int(meta["resolved_control_step"]))
    for _ in range(remaining):
     if not queue:
      if replans==0: noise=first_noise
      else:
       batch=prepare(policy,pre,env_pre,obs,meta["instruction"]); g=torch.Generator(device=batch["state"].device).manual_seed(int(spec["noise_seed"])+replans); noise=torch.randn((1,cfg.chunk_size,cfg.max_action_dim),generator=g,device=batch["state"].device,dtype=batch["state"].dtype)
      all_noise.append(digest(noise))
      with torch.inference_mode():
       if arm!="V" and replans<3:
        chunk,trace=sample_signed_region_actions(policy.model,batch["images"],batch["image_masks"],batch["lang_tokens"],batch["lang_masks"],batch["state"],noise,spec["candidate"],visual,cameras,direction=spec["direction"]); intervention.append({"replan":replans,**trace})
       else: chunk=policy.model.sample_actions(batch["images"],batch["image_masks"],batch["lang_tokens"],batch["lang_masks"],batch["state"],noise=noise)
      if not bool(torch.isfinite(chunk).all()): raise RuntimeError("nonfinite chunk")
      queue=[x.detach().cpu() for x in chunk[:,:10,:7].transpose(0,1)]; replans+=1
     ma=queue.pop(0); legal=env_post({"action":post(ma)})["action"]; obs,_,terminated,_,info=env.step(legal.detach().cpu().numpy()); actions.append(ma[0].detach().float().cpu()); success=bool(vector_info_value(info,"is_success"))
     if success: reason="success"; break
     if bool(terminated[0]): reason="terminated"; break
   finally: env.close()
   expected_interventions=min(3,replans) if arm!="V" else 0
   records[arm]={**spec,"status":"complete","success":success,"termination_reason":reason,"continuation_control_steps":len(actions),"replans":replans,"candidate_replans_applied":len(intervention),"expected_candidate_replans_given_episode_length":expected_interventions,"intervention_traces":intervention,"noise_sha256_by_replan":all_noise,"all_actions_finite":all(bool(torch.isfinite(x).all()) for x in actions),"branch_point":branch[arm],"protocol_sha256":protocol_sha,"manifest_sha256":manifest_sha,"mean_applied_correction_norm":float(np.mean([s["applied_correction_norm"] for t in intervention for s in t["steps"]])) if intervention else 0.0}
  for field in ("fingerprints","noise_sha256","clean_chunk_sha256"):
   if len({json.dumps(branch[x][field],sort_keys=True) for x in ARMS})!=1: invalid=field; break
  if invalid is None and not all(branch[x]["reference_equal"] for x in ARMS): invalid="reference_fingerprint"
  if invalid is None:
   for arm in ARMS:
    r=records[arm]
    if not r["all_actions_finite"] or r["candidate_replans_applied"]!=r["expected_candidate_replans_given_episode_length"]: invalid=f"{arm}_duration_or_finite"; break
    if arm!="V" and any(len(t["changed_indices"])!=4 or not t["protected_tokens_untouched"] for t in r["intervention_traces"]): invalid=f"{arm}_token_integrity"; break
  if invalid:
   atomic(art/"invalid_units"/f"{unit_id}.json",{"unit_id":unit_id,"first_mismatch":invalid,"branch":branch}); raise RuntimeError(f"invalid {unit_id}: {invalid}")
  if a.dry_run:
   atomic(art/"gate"/"dryrun.json",{"decision":"TOKEN_DIRECTION_ORACLE_U0_DRYRUN_PASS","unit_id":unit_id,"outcomes_not_reported":True,"branch_hashes_match":True,"duration_and_token_integrity":True}); atomic(art/"decision.json",{"decision":"TOKEN_DIRECTION_ORACLE_U0_DRYRUN_PASS"}); print("TOKEN_DIRECTION_ORACLE_U0_DRYRUN_PASS"); return
  for arm in ARMS: atomic(art/"episodes"/f"{specs[arm]['episode_id']}.json",records[arm])
  complete+=1; print(json.dumps({"worker":a.unit_start,"complete":complete,"total":len(ordered),"unit_id":unit_id}),flush=True)
 (art/"status"/f"worker_{a.unit_start:02d}.complete").write_text(f"{complete}/{len(ordered)}\n")

if __name__=="__main__": main()
