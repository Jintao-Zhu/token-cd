#!/usr/bin/env python3
"""Lock task-8 signed token causal-calibration protocol and run its smoke gate."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib,torch,yaml
from research.coreact_closed_loop.runtime import load_policy_and_processors,make_task_env,prepare
from research.coreact_value_probe.token_utility import signed_token_sets

CONDITIONS=("vanilla","mask_strong_positive","mask_strong_negative","mask_near_zero")
LANGUAGE="pick up the black bowl next to the plate and place it on the plate"
def main():
 p=argparse.ArgumentParser();p.add_argument("--workspace",type=Path,required=True);p.add_argument("--artifact",type=Path,required=True);a=p.parse_args();w,o=a.workspace.resolve(),a.artifact.resolve();o.mkdir(parents=True,exist_ok=False);(o/"episodes").mkdir();(o/"logs").mkdir()
 protocol={"experiment":"success_value_token_causal_calibration_v1","suite":"libero_spatial","task_id":8,"task_split":"never used in value-probe train/validation/test","init_states":"0-49","conditions":list(CONDITIONS),"proposal":"late-half action-to-context attention top-32 at tau=1","utility":"linear V(clean)-V(single-token-position-mean-mask)","groups":{"size":8,"positive":"largest utility","negative":"smallest utility","neutral":"smallest absolute utility, disjoint from other sets"},"intervention":"masked-only prefix for first 3 replans, then vanilla","shared":"init state, reset seed, per-replan Gaussian noise, preprocessing, 10-step solver, H50 execute10","maximum_steps":280,"analysis":"paired success differences; episode bootstrap 2000; signed utility versus mask harm","go_gate":"positive mask worse than vanilla and negative mask better than positive with CI directions; neutral within 5pp of vanilla","guidance":False}
 (o/"protocol.lock.yaml").write_text(yaml.safe_dump(protocol,sort_keys=False))
 rows=[]
 for init in range(50):
  for c in CONDITIONS:rows.append({"episode_id":f"task08__init{init:02d}__{c}","pair_id":f"task08__init{init:02d}","suite":"libero_spatial","task_id":8,"init_state_id":init,"condition":c,"language":LANGUAGE,"reset_seed":88000000+init,"action_noise_seed":88100000+init})
 with (o/"episode_manifest.jsonl").open("x") as f:
  for r in rows:f.write(json.dumps(r,sort_keys=True)+"\n")
 cfg,policy,pre,_=load_policy_and_processors(w);probe=joblib.load(w/"artifacts/coreact_success_value_probe_v1_20260808_145958/linear_probe.joblib");means=torch.load(w/"artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt",weights_only=True,map_location="cpu")["visual_position_mean"]
 env,env_pre,_=make_task_env("libero_spatial",8,cfg)
 try:
  env.envs[0].init_state_id=0;obs,_=env.reset(seed=88000000);p0=prepare(policy,pre,env_pre,obs,LANGUAGE);g=torch.Generator(device=p0["state"].device).manual_seed(88100000*1000);noise=torch.randn((1,cfg.chunk_size,cfg.max_action_dim),generator=g,device=p0["state"].device);a1=signed_token_sets(policy.model,p0,noise,means,probe);a2=signed_token_sets(policy.model,p0,noise,means,probe)
 finally:env.close()
 sets=a1["sets"];flat=sum(sets.values(),[]);checks={"frozen_eval":not policy.training and not any(x.requires_grad for x in policy.parameters()),"deterministic":sets==a2["sets"] and a1["candidate_utilities"]==a2["candidate_utilities"],"top32":len(a1["candidate_indices"])==32,"groups_8_disjoint":all(len(v)==8 for v in sets.values()) and len(set(flat))==24,"finite":bool(torch.isfinite(torch.tensor(a1["candidate_utilities"])).all())}
 report={"pass":all(checks.values()),"checks":checks,"clean_value":a1["clean_value"],"sets":sets,"candidate_utilities":a1["candidate_utilities"],"prefix_sha256":a1["prefix_sha256"]};(o/"integrity_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 if not report["pass"]:raise SystemExit(1)
 print(json.dumps({"artifact":str(o),"episodes":len(rows),"integrity":report},indent=2))
if __name__=="__main__":main()
