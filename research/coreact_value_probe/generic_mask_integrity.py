#!/usr/bin/env python3
"""Run the pre-rollout native/masked determinism gate on all three new tasks."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from research.coreact_closed_loop.guidance import GuidanceConfig,tensor_sha256
from research.coreact_closed_loop.runtime import load_policy_and_processors,make_task_env,prepare
from research.coreact_revision.masked_sampler import sample_masked_actions
def main():
 p=argparse.ArgumentParser();p.add_argument("--workspace",type=Path,required=True);p.add_argument("--artifact",type=Path,required=True);a=p.parse_args();w,o=a.workspace.resolve(),a.artifact.resolve();cfg,policy,pre,_=load_policy_and_processors(w);means=torch.load(w/"artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt",weights_only=True,map_location="cpu")["visual_position_mean"];checks={};
 for tid,lang in zip((0,1,2),("open the middle drawer of the cabinet","put the bowl on the stove","put the wine bottle on top of the cabinet")):
  env,ep,_=make_task_env("libero_goal",tid,cfg)
  try:
   env.envs[0].init_state_id=0;obs,_=env.reset(seed=99000000+tid);b=prepare(policy,pre,ep,obs,lang);g=torch.Generator(device=b["state"].device).manual_seed((99100000+tid)*1000);noise=torch.randn((1,cfg.chunk_size,cfg.max_action_dim),generator=g,device=b["state"].device,dtype=b["state"].dtype)
   with torch.inference_mode():
    n1=policy.model.sample_actions(b["images"],b["image_masks"],b["lang_tokens"],b["lang_masks"],b["state"],noise=noise);n2=policy.model.sample_actions(b["images"],b["image_masks"],b["lang_tokens"],b["lang_masks"],b["state"],noise=noise);outs={}
    for name,mode in (("top","top"),("random","random"),("bottom","bottom")):
     m1,t1=sample_masked_actions(policy.model,b["images"],b["image_masks"],b["lang_tokens"],b["lang_masks"],b["state"],noise,means,config=GuidanceConfig(selection=mode,group_count=8,action_dim=7,num_steps=10),selection_seed=99200000+tid);m2,t2=sample_masked_actions(policy.model,b["images"],b["image_masks"],b["lang_tokens"],b["lang_masks"],b["state"],noise,means,config=GuidanceConfig(selection=mode,group_count=8,action_dim=7,num_steps=10),selection_seed=99200000+tid);outs[name]={"native_diff":float((n1-n2).abs().max()),"masked_diff":float((m1-m2).abs().max()),"changed":len(t1["changed_indices"]),"selected_equal_changed":sorted(t1["selected_indices"])==sorted(t1["changed_indices"]),"repeatable":t1["selected_indices"]==t2["selected_indices"] and t1["masked_prefix_sha256"]==t2["masked_prefix_sha256"],"finite":bool(torch.isfinite(m1).all()),"protected":t1["protected_tokens_untouched"]}
   checks[str(tid)]={"state_input_dim":int(b["batch"]["observation.state"].shape[-1]),"model_state_dim":int(b["state"].shape[-1]),"conditions":outs}
  finally:env.close()
 passed=not policy.training and not any(x.requires_grad for x in policy.parameters()) and all(v["state_input_dim"]==8 and v["model_state_dim"]==32 and all(z["native_diff"]<=1e-6 and z["masked_diff"]<=1e-6 and z["changed"]==8 and z["selected_equal_changed"] and z["repeatable"] and z["finite"] and z["protected"] for z in v["conditions"].values()) for v in checks.values());r={"gate":"generic_masking_integrity","pass":passed,"tasks":checks};(o/"integrity_report.json").write_text(json.dumps(r,indent=2,sort_keys=True)+"\n");print(json.dumps(r,indent=2));
 if not passed:raise SystemExit(1)
if __name__=="__main__":main()
