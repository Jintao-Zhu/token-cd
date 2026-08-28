#!/usr/bin/env python3
"""Mandatory real-state integrity gate for the four-condition direction experiment."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from research.coreact_closed_loop.guidance import GuidanceConfig,sample_coreact_actions,tensor_sha256
from research.coreact_closed_loop.runtime import load_policy_and_processors,make_task_env,prepare
from research.coreact_revision.consistency_guidance import sample_consistency_guided_actions
from research.coreact_revision.run_online_consistency_qualification import fit_task_heldout_classifier
def main():
 p=argparse.ArgumentParser();p.add_argument("--workspace",type=Path,required=True);p.add_argument("--artifact",type=Path,required=True);a=p.parse_args();w,art=a.workspace.resolve(),a.artifact.resolve();spec=json.loads((art/"episode_manifest.jsonl").read_text().splitlines()[0]);source=w/"artifacts/coreact_region_sign_multitask_v1_20260808_093747";abl=w/"artifacts/coreact_action_consistency_ablation_v1_20260808_113116"
 clf,_,numeric=fit_task_heldout_classifier(source/"raw_effects.jsonl",abl/"consistency_scores.jsonl",heldout_task=4);cfg,policy,pre,_=load_policy_and_processors(w);means=torch.load(w/"artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt",weights_only=True,map_location="cpu");env,env_pre,_=make_task_env("libero_spatial",4,cfg)
 try:
  env.envs[0].init_state_id=0;obs,_=env.reset(seed=spec["reset_seed"]);x=prepare(policy,pre,env_pre,obs,spec["language"]);noise=torch.randn((1,cfg.chunk_size,cfg.max_action_dim),generator=torch.Generator(device=x["state"].device).manual_seed(spec["action_noise_seed"]*1000),device=x["state"].device,dtype=x["state"].dtype);g=GuidanceConfig(group_count=8,guidance_scale=.5,trust_region_kappa=.25,action_dim=7,num_steps=10,direction="toward")
  def consistency(mode):return sample_consistency_guided_actions(policy.model,x["images"],x["image_masks"],x["lang_tokens"],x["lang_masks"],x["state"],noise,means,clf,numeric,config=g,mode=mode,nuisance_threshold=.5)
  with torch.inference_mode():
   va=policy.model.sample_actions(x["images"],x["image_masks"],x["lang_tokens"],x["lang_masks"],x["state"],noise=noise);vb=policy.model.sample_actions(x["images"],x["image_masks"],x["lang_tokens"],x["lang_masks"],x["state"],noise=noise)
   aa,at=sample_coreact_actions(policy.model,x["images"],x["image_masks"],x["lang_tokens"],x["lang_masks"],x["state"],noise,means["visual_position_mean"],config=g,selection_seed=0);ab,abt=sample_coreact_actions(policy.model,x["images"],x["image_masks"],x["lang_tokens"],x["lang_masks"],x["state"],noise,means["visual_position_mean"],config=g,selection_seed=0)
   ma,mt=consistency("mask_only");mb,mbt=consistency("mask_only");ta,tt=consistency("toward");tb,tbt=consistency("toward")
  checks={"policy_eval":not policy.training,"parameters_frozen":not any(p.requires_grad for p in policy.parameters()),"vanilla_repeat_max_abs":float((va-vb).abs().max()),"attention_repeat_max_abs":float((aa-ab).abs().max()),"mask_repeat_max_abs":float((ma-mb).abs().max()),"toward_repeat_max_abs":float((ta-tb).abs().max()),"attention_selected_8":len(at["selected_indices"])==8,"consistency_selected_at_most_8":0<=mt["selected_count"]<=8,"consistency_selection_repeatable":mt["selected_indices"]==mbt["selected_indices"]==tt["selected_indices"]==tbt["selected_indices"],"consistency_masked_prefix_identical":mt["negative_prefix_sha256"]==mbt["negative_prefix_sha256"]==tt["negative_prefix_sha256"]==tbt["negative_prefix_sha256"],"selected_equals_changed":sorted(mt["selected_indices"])==sorted(mt["changed_indices"]) and sorted(tt["selected_indices"])==sorted(tt["changed_indices"]),"threshold_respected":all(v>=.5 for v in mt["selected_probabilities"]),"protected_untouched":at["protected_tokens_untouched"] and mt["protected_tokens_untouched"] and tt["protected_tokens_untouched"],"all_finite":all(bool(torch.isfinite(v).all()) for v in (va,aa,ma,ta)) and at["all_output_finite"] and mt["all_output_finite"] and tt["all_output_finite"],"shared_noise_sha256":tensor_sha256(noise),"attention_layers":at["attention_layer_count"]==mt["attention_layer_count"]==tt["attention_layer_count"]==16}
  passed=all(v<=1e-6 if k.endswith("repeat_max_abs") else bool(v) for k,v in checks.items() if k!="shared_noise_sha256");report={"gate":"four_condition_consistency_direction_integrity","pass":passed,"checks":checks,"attention_selected":at["selected_indices"],"consistency_selected":mt["selected_indices"],"consistency_probabilities":mt["selected_probabilities"],"output_hashes":{"vanilla":tensor_sha256(va),"attention_toward":tensor_sha256(aa),"consistency_mask_only":tensor_sha256(ma),"consistency_toward":tensor_sha256(ta)}};(art/"integrity_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report,indent=2,sort_keys=True));
  if not passed:raise SystemExit(1)
 finally:env.close()
if __name__=="__main__":main()
