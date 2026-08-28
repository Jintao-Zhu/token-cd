#!/usr/bin/env python3
"""Real-state sign and parity gate for away/toward guidance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from research.coreact_closed_loop.guidance import GuidanceConfig, sample_coreact_actions, tensor_sha256
from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare
from research.coreact_revision.masked_sampler import sample_masked_actions
from research.coreact_task4_replication.run import prepared_hash


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--workspace",type=Path,required=True); p.add_argument("--artifact",type=Path,required=True); args=p.parse_args()
    workspace=args.workspace.resolve(); artifact=args.artifact.resolve(); spec=json.loads((artifact/"episode_manifest.jsonl").read_text().splitlines()[0]); protocol=yaml.safe_load((artifact/"protocol.lock.yaml").read_text()); top_k=int(protocol["selection"]["top_k"])
    config,policy,pre,_=load_policy_and_processors(workspace); means=torch.load(workspace/"artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt",weights_only=True,map_location="cpu")
    env,ep,_=make_task_env(spec["suite"],spec["task_id"],config)
    try:
        env.envs[0].init_state_id=spec["init_state_id"]; env.reset(seed=spec["reset_seed"]); obs,_=env.reset(seed=spec["reset_seed"])
        batch=prepare(policy,pre,ep,obs,spec["language"]); gen=torch.Generator(device=batch["state"].device).manual_seed(spec["action_noise_seed"]*1000)
        noise=torch.randn((1,config.chunk_size,config.max_action_dim),generator=gen,dtype=batch["state"].dtype,device=batch["state"].device)
        common=dict(selection="top",group_count=top_k,guidance_scale=0.5,trust_region_kappa=0.25,action_dim=7,num_steps=10)
        away_cfg=GuidanceConfig(**common,direction="away"); toward_cfg=GuidanceConfig(**common,direction="toward")
        def away(): return sample_coreact_actions(policy.model,batch["images"],batch["image_masks"],batch["lang_tokens"],batch["lang_masks"],batch["state"],noise,means["visual_position_mean"],config=away_cfg,selection_seed=spec["selection_seed"]*1000)
        def toward(): return sample_coreact_actions(policy.model,batch["images"],batch["image_masks"],batch["lang_tokens"],batch["lang_masks"],batch["state"],noise,means["visual_position_mean"],config=toward_cfg,selection_seed=spec["selection_seed"]*1000)
        def masked(): return sample_masked_actions(policy.model,batch["images"],batch["image_masks"],batch["lang_tokens"],batch["lang_masks"],batch["state"],noise,means["visual_position_mean"],config=away_cfg,selection_seed=spec["selection_seed"]*1000)
        with torch.inference_mode():
            native1=policy.model.sample_actions(batch["images"],batch["image_masks"],batch["lang_tokens"],batch["lang_masks"],batch["state"],noise=noise); native2=policy.model.sample_actions(batch["images"],batch["image_masks"],batch["lang_tokens"],batch["lang_masks"],batch["state"],noise=noise)
            away1,at1=away(); away2,at2=away(); toward1,tt1=toward(); toward2,tt2=toward(); mask1,mt1=masked(); mask2,mt2=masked()
        a0=at1["step_traces"][0]; t0=tt1["step_traces"][0]
        checks={"policy_eval":not policy.training,"all_parameters_frozen":not any(x.requires_grad for x in policy.parameters()),
            "native_repeat_max_abs":float((native1-native2).abs().max()),"away_repeat_max_abs":float((away1-away2).abs().max()),
            "toward_repeat_max_abs":float((toward1-toward2).abs().max()),"mask_repeat_max_abs":float((mask1-mask2).abs().max()),
            "all_top8_identical":at1["selected_indices"]==tt1["selected_indices"]==mt1["selected_indices"],
            "exactly_locked_top_k_changed":len(at1["changed_indices"])==len(tt1["changed_indices"])==len(mt1["changed_indices"])==top_k,
            "native_prefix_identical":at1["prefix_sha256"]==tt1["prefix_sha256"]==mt1["prefix_sha256"],
            "masked_prefix_identical":at1["negative_prefix_sha256"]==tt1["negative_prefix_sha256"]==mt1["masked_prefix_sha256"],
            "first_step_positive_velocity_identical":a0["positive_velocity_sha256"]==t0["positive_velocity_sha256"],
            "first_step_raw_delta_identical":a0["raw_guidance_sha256"]==t0["raw_guidance_sha256"],
            "first_step_guidance_exact_opposites":abs(a0["signed_guidance_sum"]+t0["signed_guidance_sum"])<=1e-6 and abs(a0["applied_guidance_norm"]-t0["applied_guidance_norm"])<=1e-6,
            "directions_explicit":at1["guidance_direction"]=="away" and tt1["guidance_direction"]=="toward",
            "protected_untouched":at1["protected_tokens_untouched"] and tt1["protected_tokens_untouched"] and mt1["protected_tokens_untouched"],
            "same_noise_sha256":tensor_sha256(noise),"all_outputs_finite":bool(torch.isfinite(torch.stack([native1,away1,toward1,mask1])).all()),
            "prepared_input_sha256":prepared_hash(batch)}
        numeric=[checks[k] for k in ("native_repeat_max_abs","away_repeat_max_abs","toward_repeat_max_abs","mask_repeat_max_abs")]
        passed=all(v for v in checks.values() if isinstance(v,bool)) and max(numeric)<=1e-6
        report={"gate":f"task{spec['task_id']}_guidance_sign_integrity","top_k":top_k,"pass":passed,"tolerance":1e-6,"checks":checks,"selected_indices":at1["selected_indices"],
            "prefix_hashes":{"native":at1["prefix_sha256"],"masked":at1["negative_prefix_sha256"]},"first_step":{"away":a0,"toward":t0},
            "output_hashes":{"native":tensor_sha256(native1),"away":tensor_sha256(away1),"toward":tensor_sha256(toward1),"mask":tensor_sha256(mask1)}}
        (artifact/"integrity_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); (artifact/"integrity_report.md").write_text("# Sign Diagnostic Integrity\n\nOverall: **"+("PASS" if passed else "FAIL")+"**\n")
        print(json.dumps(report,indent=2,sort_keys=True)); raise SystemExit(0 if passed else 1)
    finally: env.close()


if __name__ == "__main__": main()
