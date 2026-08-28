from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare
from research.coreact_self_guidance.relative_sampler import sample_relative_self_guided_actions
from research.coreact_self_guidance.sampler import select_ref_top8_actions


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--workspace", type=Path, required=True); parser.add_argument("--artifact", type=Path, required=True); args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    if (artifact / "phase0_calibration.json").exists(): raise FileExistsError("refusing to overwrite phase0 calibration")
    protocol = yaml.safe_load((artifact / "protocol.phase0.yaml").read_text())
    config, policy, preprocessor, _ = load_policy_and_processors(workspace)
    means = torch.load(workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt", weights_only=True, map_location="cpu")
    states = [json.loads(line) for line in (artifact / "phase0_state_manifest.jsonl").read_text().splitlines()]
    values = {float(alpha): [] for alpha in protocol["alpha_grid"]}; raw_values = {float(alpha): [] for alpha in protocol["alpha_grid"]}; active = {float(alpha): [] for alpha in protocol["alpha_grid"]}; clips = {float(alpha): [] for alpha in protocol["alpha_grid"]}; state_hashes = []; same_state_reference=[]
    for spec in states:
        env, env_preprocessor, _ = make_task_env("libero_spatial", spec["task_id"], config)
        try:
            env.envs[0].init_state_id = spec["init_state_id"]; observation, _ = env.reset(seed=spec["reset_seed"]); batch = prepare(policy, preprocessor, env_preprocessor, observation, spec["language"])
            tensors = [*batch["images"], batch["state"], batch["lang_tokens"]]; state_hashes.append({"state_id": spec["state_id"], "input_sha256": hashlib.sha256(b"".join(t.detach().cpu().contiguous().numpy().tobytes() for t in tensors)).hexdigest()})
            generator = torch.Generator(device=batch["state"].device).manual_seed(spec["action_noise_seed"] * 1000); noise = torch.randn((1, config.chunk_size, config.max_action_dim), generator=generator, device=batch["state"].device, dtype=batch["state"].dtype)
            _, reference_trace = select_ref_top8_actions(policy.model, batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"], noise, means, selection_seed=spec["action_noise_seed"] + 17)
            same_state_reference.extend(s["applied_guidance_norm"] for s in reference_trace["step_traces"])
            for alpha in values:
                _, trace = sample_relative_self_guided_actions(policy.model, batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"], noise, alpha=alpha, w=0.5)
                steps = trace["step_traces"]; post = [s["applied_delta_l2_norm_post_clip"] for s in steps]; raw = [s["clean_minus_v_neg_l2_norm_pre_clip"] for s in steps]
                values[alpha].extend(post); raw_values[alpha].extend(raw); active[alpha].extend([s["active_earlier_self_bool"] for s in steps]); clips[alpha].extend([s["trust_region_clipping_active_bool"] for s in steps])
        finally: env.close()
    e1 = workspace / "artifacts/coreact_ensemble_vs_contrast_control_v1_20260809_152414"; target=[]
    for path in e1.glob("episodes/*.json"):
        record=json.loads(path.read_text())
        if record["arm"] == "B_toward_top8":
            for trace in record["replan_traces"]: target.extend(trace["applied_delta_l2_norm_post_clip"])
    if not target: raise RuntimeError("E1 post-clip reference missing")
    target_median=float(np.median(target)); target_mean=float(np.mean(target)); target_rms=float(np.sqrt(np.mean(np.square(target))))
    same_state_target_median=float(np.median(same_state_reference)); same_state_target_mean=float(np.mean(same_state_reference)); same_state_target_rms=float(np.sqrt(np.mean(np.square(same_state_reference))))
    candidates=[]
    for alpha in values:
        post=np.asarray(values[alpha]); raw=np.asarray(raw_values[alpha]); active_fraction=float(np.mean(active[alpha])); clip_fraction=float(np.mean(clips[alpha]));
        candidates.append({"alpha":alpha,"post_clip_median":float(np.median(post)),"post_clip_mean":float(np.mean(post)),"post_clip_rms":float(np.sqrt(np.mean(post**2))),"post_clip_integrated_l2_per_trajectory":float(np.sum(post)/len(states)),"raw_mean":float(np.mean(raw)),"active_step_fraction":active_fraction,"clip_fraction":clip_fraction,"ratio_to_same_state_ref_median":float(np.median(post)/(same_state_target_median+1e-12)),"ratio_to_e1_post_clip_median":float(np.median(post)/(target_median+1e-12)),"n_steps":len(post)})
    legal=[c for c in candidates if c["active_step_fraction"] >= 0.8 and 0.7 <= c["ratio_to_same_state_ref_median"] <= 1.4]
    if not legal:
        result={"pass":False,"reason":"no alpha matched trajectory post-clip ratio and active-step gate","target":{"same_state_ref":{"median":same_state_target_median,"mean":same_state_target_mean,"rms":same_state_target_rms},"e1_global":{"median":target_median,"mean":target_mean,"rms":target_rms}},"candidates":candidates,"state_hashes":state_hashes,"outcome_blind":True}; (artifact/"phase0_calibration.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); raise RuntimeError("relative trajectory calibration failed")
    chosen=min(legal,key=lambda c:(abs(c["ratio_to_same_state_ref_median"]-1),c["alpha"]))
    result={"pass":True,"reference_source":str(e1),"same_state_reference":"phase0 REF_toward_top8 computed on identical states/noise","target":{"same_state_ref":{"median":same_state_target_median,"mean":same_state_target_mean,"rms":same_state_target_rms,"n":len(same_state_reference)},"e1_global":{"median":target_median,"mean":target_mean,"rms":target_rms,"n":len(target)}},"candidates":candidates,"selected":chosen,"state_hashes":state_hashes,"outcome_blind":True}
    (artifact/"phase0_calibration.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    protocol["locked_alpha"]=chosen["alpha"]; protocol["phase0_result"]={"selected_alpha":chosen["alpha"],"trajectory_target_post_clip_median":same_state_target_median,"selected_ratio":chosen["ratio_to_same_state_ref_median"],"active_step_fraction":chosen["active_step_fraction"],"outcome_blind":True}; (artifact/"protocol.lock.yaml").write_text(yaml.safe_dump(protocol,sort_keys=False),encoding="utf-8")
    bases={}
    for row in [json.loads(x) for x in (e1/"episode_manifest.jsonl").read_text().splitlines()]:
        if row["arm"]=="A_vanilla": bases[(row["task_id"],row["init_state_id"])] = row
    arms=("A_vanilla","N0_pure_negative","W05_shrink","W15_extrapolate","W20_extrapolate","REF_toward_top8"); ws={"A_vanilla":1.0,"N0_pure_negative":0.0,"W05_shrink":0.5,"W15_extrapolate":1.5,"W20_extrapolate":2.0,"REF_toward_top8":0.5}; manifest=[]
    for task_id in (4,7):
        for init_state_id in range(50):
            base=bases[(task_id,init_state_id)]
            for arm in arms: manifest.append({**base,"episode_id":f"task{task_id:02d}__init{init_state_id:02d}__{arm}","arm":arm,"alpha":chosen["alpha"],"w":ws[arm]})
    with (artifact/"episode_manifest.jsonl").open("x") as stream:
        for row in manifest: stream.write(json.dumps(row,sort_keys=True)+"\n")
    print(json.dumps({"phase0":result,"episodes":len(manifest),"locked_alpha":chosen["alpha"]},indent=2))


if __name__ == "__main__":main()
