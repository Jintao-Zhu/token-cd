from __future__ import annotations

import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import torch, yaml

from research.coreact_closed_loop.guidance import _full_velocity, tensor_sha256
from research.coreact_closed_loop.runtime import load_policy_and_processors
from research.coreact_exploration.instrumentation import attention_ranking_scores, build_prefix_span_map
from research.coreact_signed_utility.token_regions import local_2x2_regions, region_features, select_oracle_candidates

ARMS = ("V", "IT", "IA", "AT", "AA", "RT", "RA")
SELECTORS = {"I": "iss_max", "A": "attention_max", "R": "random_control"}

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--workspace",type=Path,required=True); p.add_argument("--reference",type=Path,required=True); p.add_argument("--means",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    ws,ref,out=a.workspace.resolve(),a.reference.resolve(),a.output.resolve()
    if out.exists(): raise FileExistsError(out)
    os.environ.setdefault("HF_HOME",str(ws/"task1/.hf-cache")); os.environ["MUJOCO_GL"]="egl"
    for d in ("candidates","features","episodes","invalid_units","gate","logs","status"): (out/d).mkdir(parents=True,exist_ok=True)
    cfg,policy,_,_=load_policy_and_processors(ws); means=torch.load(a.means,weights_only=False,map_location="cpu")
    visual=means["visual_position_mean"].to(cfg.device,dtype=torch.float32); cameras=tuple(means["camera_ids"])
    snapshots=sorted((ref/"snapshots").glob("*.pt"))
    if len(snapshots)!=40: raise RuntimeError(f"expected 40 snapshots, got {len(snapshots)}")
    protocol={"experiment_name":"coreact_token_direction_oracle_headroom_u0_v1","stage":"locked_before_outcomes",
      "question":"held-out snapshot-level oracle headroom among existing token-induced directions",
      "suite":"libero_spatial","tasks":list(range(10)),"fresh_init_state_ids":[10,11],"progress_points":[0.25,0.65],"snapshots":40,
      "arms":list(ARMS),"arm_definitions":{"V":"Vanilla","IT":"ISS Toward","IA":"ISS Away","AT":"Attention Toward","AA":"Attention Away","RT":"Random Toward","RA":"Random Away"},
      "candidate_definitions":{"iss_max":"T0 maximum full 10-step flow action RMS, non-overlapping after Attention","attention_max":"T0 maximum late-half action-to-context attention","random_control":"T0 deterministic random remaining non-overlapping region"},
      "replacement":"T0 camera-and-position-conditioned visual embedding mean","region":"T0 local 2x2 connector-token region (4 tokens)",
      "lambda":0.5,"trust_region_kappa":0.25,"candidate_replans":3,"later_replans":"Vanilla","selection_seeds":[0,1,2],"evaluation_seeds":[3,4,5],
      "noise_rule":"202608130000 + task*100000 + init*10000 + bin*1000 + seed","planned_episodes":1680,"paired_evaluation_units":120,
      "tie_rule":["Vanilla if tied at maximum","otherwise smaller mean applied correction norm on selection seeds","then fixed order V,IT,IA,AT,AA,RT,RA"],
      "primary":"evaluation-only OracleSelect minus Vanilla success rate","bootstrap_units":["snapshot cluster","task stratified snapshot cluster"],
      "strong_go":{"delta_pp_min":8,"ci_lower_gt_zero":True,"nonnegative_tasks_min":7,"rescue_gt_harm":True,"max_task_harm_pp":15,"oracle_over_best_fixed_pp_min_or_less_heterogeneous":3},
      "promising":{"delta_pp_range":[5,8],"nonnegative_tasks_min":7,"rescue_gt_harm":True,"fixed_arm_same_gain_forbidden":True},
      "no_go":["delta_pp_le_3","oracle approximately best fixed","task signs heterogeneous","gain concentrated in one or two tasks","rescue approximately harm"],
      "forbidden":["training","new token score","lambda search","partial aggregate outcomes","outcome-based snapshot filtering"],
      "reference":str(ref),"reference_lock_sha256":sha(ref/"reference_gate.lock.yaml"),"means_sha256":sha(a.means.resolve())}
    (out/"protocol.lock.yaml").write_text(yaml.safe_dump(protocol,sort_keys=False))
    candidates=[]; episodes=[]
    for path in snapshots:
      snap=torch.load(path,weights_only=False,map_location="cpu"); meta=snap["metadata"]; prepared=snap["reference_prepared"]; device=torch.device(cfg.device)
      images=[x.to(device) for x in prepared["images"]]; masks=[x.to(device) for x in prepared["image_masks"]]; lt=prepared["lang_tokens"].to(device); lm=prepared["lang_masks"].to(device); state=prepared["state"].to(device)
      prefix,pads,atts=policy.model.embed_prefix(images,masks,lt,lm,state=state); span=build_prefix_span_map(policy.model,images,masks,lt,lm,pads,camera_ids=cameras)[0]; regions=local_2x2_regions(span,cameras)
      bin_id=int(meta["target_progress"]>0.5); probe=202608139000+meta["task_id"]*100+meta["init_state_id"]*10+bin_id; g=torch.Generator(device=device).manual_seed(probe); noise=torch.randn((1,cfg.chunk_size,cfg.max_action_dim),generator=g,device=device,dtype=state.dtype)
      _,traces=_full_velocity(policy.model,prefix,pads,atts,noise,torch.ones(1,device=device),record_attention=True); scores=attention_ranking_scores(traces,prefix.shape[1])["late_half_action_to_context_attention"]
      rows,_=region_features(policy.model,prefix,pads,atts,span,regions,scores,noise,visual,cameras); chosen=select_oracle_candidates(rows,probe); sid=meta["snapshot_id"]
      record={"snapshot_id":sid,"snapshot_file":str(path),"task_id":meta["task_id"],"init_state_id":meta["init_state_id"],"target_progress":meta["target_progress"],"reference_fingerprints":meta["reference_fingerprints"],"probe_seed":probe,"probe_noise_sha256":tensor_sha256(noise),"prefix_sha256":tensor_sha256(prefix),"candidates":chosen}
      dump(out/"candidates"/f"{sid}.json",record); torch.save({"state":record,"all_region_features":rows},out/"features"/f"{sid}.pt"); candidates.append(record); byid={x["candidate_id"]:x for x in chosen}
      for seed in range(6):
       noise_seed=202608130000+meta["task_id"]*100000+meta["init_state_id"]*10000+bin_id*1000+seed
       for arm in ARMS:
        candidate=None if arm=="V" else byid[SELECTORS[arm[0]]]
        episodes.append({"episode_id":f"{sid}__seed{seed:02d}__{arm}","unit_id":f"{sid}__seed{seed:02d}","snapshot_id":sid,"task_id":meta["task_id"],"init_state_id":meta["init_state_id"],"target_progress":meta["target_progress"],"seed_role":"selection" if seed<3 else "evaluation","continuation_seed":seed,"noise_seed":noise_seed,"arm":arm,"candidate":candidate,"direction":None if arm=="V" else ("toward" if arm[1]=="T" else "away")})
      print(json.dumps({"prepared":len(candidates),"snapshot_id":sid}),flush=True)
    (out/"candidate_manifest.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in candidates)); (out/"episode_manifest.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in episodes))
    checks={"snapshots":len(candidates),"episodes":len(episodes),"three_candidates":all(len(x["candidates"])==3 for x in candidates),"unique_regions":all(len({z["region_id"] for z in x["candidates"]})==3 for x in candidates),"finite":all(z["finite"] for x in candidates for z in x["candidates"])}; checks["pass"]=checks=={**checks,"pass":True} if False else all([checks["snapshots"]==40,checks["episodes"]==1680,checks["three_candidates"],checks["unique_regions"],checks["finite"]])
    checks.update({"protocol_sha256":sha(out/"protocol.lock.yaml"),"candidate_manifest_sha256":sha(out/"candidate_manifest.jsonl"),"episode_manifest_sha256":sha(out/"episode_manifest.jsonl")}); dump(out/"candidate_gate.json",checks); decision="TOKEN_DIRECTION_ORACLE_U0_CANDIDATES_LOCKED" if checks["pass"] else "TOKEN_DIRECTION_ORACLE_U0_CANDIDATE_GATE_FAILED"; dump(out/"decision.json",{"decision":decision,**checks}); print(decision)

if __name__=="__main__": main()
