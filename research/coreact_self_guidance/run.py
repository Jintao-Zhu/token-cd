from __future__ import annotations

import argparse
import copy
import json
import math
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from research.coreact_closed_loop.guidance import GuidanceConfig, sample_coreact_actions, tensor_sha256
from research.coreact_closed_loop.run_pilot import vector_info_value
from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare
from research.coreact_self_guidance.sampler import sample_self_guided_actions
from research.coreact_self_guidance.relative_sampler import sample_relative_self_guided_actions
from research.coreact_self_guidance.timestep_sampler import sample_timestep_shift_actions


ARMS = {"A_vanilla", "N0_pure_negative", "W05_shrink", "W15_extrapolate", "W20_extrapolate", "REF_toward_top8"}


def array_sha256(value):
    value = np.ascontiguousarray(value); digest = hashlib.sha256(); digest.update(str(value.dtype).encode()); digest.update(str(value.shape).encode()); digest.update(value.tobytes()); return digest.hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepared_sha256(batch):
    values = [*batch["images"], *batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"]]
    return hashlib.sha256("".join(tensor_sha256(value) for value in values).encode("ascii")).hexdigest()


def select_pair_shard_rows(manifest, shard_index, shard_count):
    """Keep every arm of a paired init state on the same worker."""
    pair_order = list(dict.fromkeys(row["pair_id"] for row in manifest))
    selected_pairs = {pair for index, pair in enumerate(pair_order) if index % shard_count == shard_index}
    return [row for row in manifest if row["pair_id"] in selected_pairs]


def load_pair_identity(artifact, rows):
    identities = {}
    for spec in rows:
        output = artifact / "episodes" / f"{spec['episode_id']}.json"
        if not output.exists():
            continue
        record = json.loads(output.read_text())
        identity = (record["initial_sim_state_sha256"], record["initial_prepared_input_sha256"])
        previous = identities.setdefault(spec["pair_id"], identity)
        if identity != previous:
            raise RuntimeError(f"existing paired input identity mismatch: {spec['pair_id']}")
    return identities


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--workspace", type=Path, required=True); parser.add_argument("--artifact", type=Path, required=True); parser.add_argument("--shard-index", type=int, required=True); parser.add_argument("--shard-count", type=int, default=6); args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    gate = json.loads((artifact/"integrity_report.json").read_text())
    if not gate.get("pass"): raise RuntimeError("integrity gate failed")
    manifest = [json.loads(line) for line in (artifact/"episode_manifest.jsonl").read_text().splitlines()]
    protocol = __import__("yaml").safe_load((artifact / "protocol.lock.yaml").read_text())
    relative_mode = "locked_alpha" in protocol
    timestep_mode = "locked_shift" in protocol
    record_vectors = bool(protocol.get("save_velocity_correction_vectors", False))
    if {row["arm"] for row in manifest} - ARMS: raise RuntimeError("unknown arm")
    rows = select_pair_shard_rows(manifest, args.shard_index, args.shard_count)
    pair_identities = load_pair_identity(artifact, rows)
    pair_initial_observations = {}
    config, policy, preprocessor, postprocessor = load_policy_and_processors(workspace)
    means = torch.load(workspace/"artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt", weights_only=True, map_location="cpu")
    toward_config = GuidanceConfig(selection="top", group_count=8, guidance_scale=0.5, trust_region_kappa=0.25, action_dim=7, num_steps=10, direction="toward", record_correction_vectors=record_vectors)
    for ordinal,spec in enumerate(rows,1):
        output = artifact/"episodes"/f"{spec['episode_id']}.json"
        if output.exists(): continue
        env, env_preprocessor, env_postprocessor = make_task_env(spec["suite"],spec["task_id"],config)
        actions=[]; queue=[]; traces=[]; latencies=[]; boundaries=[]; first_noise_hashes=[]; correction_vectors=[]; applied_correction_vectors=[]; clean_velocities=[]; shifted_velocities=[]; flow_states=[]; replans=0; success=False
        try:
            inner=env.envs[0]; inner.init_state_id=spec["init_state_id"]; observation,_=env.reset(seed=spec["reset_seed"]); initial_state_hash=array_sha256(np.asarray(inner._env.get_sim_state())); initial_prepared_hash=None; torch.cuda.reset_peak_memory_stats()
            canonical_observation = pair_initial_observations.setdefault(spec["pair_id"], copy.deepcopy(observation))
            observation = copy.deepcopy(canonical_observation)
            for _step in range(280):
                if not queue:
                    batch=prepare(policy,preprocessor,env_preprocessor,observation,spec["language"])
                    if initial_prepared_hash is None:
                        initial_prepared_hash=prepared_sha256(batch)
                        identity=(initial_state_hash, initial_prepared_hash)
                        canonical=pair_identities.setdefault(spec["pair_id"], identity)
                        if identity != canonical:
                            raise RuntimeError(
                                f"paired input identity mismatch before action generation: {spec['pair_id']} "
                                f"expected={canonical} observed={identity}"
                            )
                    shape=(1,config.chunk_size,config.max_action_dim); gen=torch.Generator(device=batch["state"].device); noise=torch.randn(shape,generator=gen.manual_seed(spec["action_noise_seed"]*1000+replans),device=batch["state"].device,dtype=batch["state"].dtype); first_noise_hashes.append(tensor_sha256(noise)); start=time.perf_counter()
                    with torch.inference_mode():
                        if spec["arm"] == "A_vanilla":
                            chunk=policy.model.sample_actions(batch["images"],batch["image_masks"],batch["lang_tokens"],batch["lang_masks"],batch["state"],noise=noise); trace={"method":"vanilla","selected_indices":[],"changed_indices":[],"noise_sha256":tensor_sha256(noise),"masked_token_count":0,"step_traces":[],"all_output_finite":bool(torch.isfinite(chunk).all()),"protected_tokens_untouched":True}
                        elif spec["arm"] == "REF_toward_top8":
                            chunk,trace=sample_coreact_actions(policy.model,batch["images"],batch["image_masks"],batch["lang_tokens"],batch["lang_masks"],batch["state"],noise,means["visual_position_mean"],config=toward_config,selection_seed=spec["selection_seed"]*1000+replans)
                        else:
                            pure=spec["arm"]=="N0_pure_negative"; w=float(spec["w"])
                            if timestep_mode:
                                chunk,trace=sample_timestep_shift_actions(policy.model,batch["images"],batch["image_masks"],batch["lang_tokens"],batch["lang_masks"],batch["state"],noise,shift=float(spec["shift"]),w=w,pure_negative=pure,record_correction_vectors=record_vectors)
                            elif relative_mode:
                                chunk,trace=sample_relative_self_guided_actions(policy.model,batch["images"],batch["image_masks"],batch["lang_tokens"],batch["lang_masks"],batch["state"],noise,alpha=float(spec["alpha"]),w=w,pure_negative=pure)
                            else:
                                chunk,trace=sample_self_guided_actions(policy.model,batch["images"],batch["image_masks"],batch["lang_tokens"],batch["lang_masks"],batch["state"],noise,delta=float(spec["delta"]),w=w,pure_negative=pure)
                    torch.cuda.synchronize(); latencies.append(time.perf_counter()-start)
                    if not torch.isfinite(chunk).all(): raise RuntimeError("nonfinite chunk")
                    prefix=chunk[:,:10,:7].transpose(0,1)
                    if actions: boundaries.append(float(torch.linalg.vector_norm(prefix[0,0].cpu()-actions[-1])))
                    raw_steps=[]; applied_steps=[]; clean_steps=[]; shifted_steps=[]; x_steps=[]
                    for step_trace in trace.get("step_traces", []):
                        if "_correction_vector" in step_trace:
                            raw_steps.append(step_trace.pop("_correction_vector")); applied_steps.append(step_trace.pop("_applied_correction_vector"))
                            if "_clean_velocity" in step_trace and "_shifted_velocity" in step_trace:
                                clean_steps.append(step_trace.pop("_clean_velocity")); shifted_steps.append(step_trace.pop("_shifted_velocity")); x_steps.append(step_trace.pop("_x_tau"))
                    if raw_steps:
                        correction_vectors.append(torch.stack(raw_steps)); applied_correction_vectors.append(torch.stack(applied_steps))
                        if len(clean_steps) != len(raw_steps) or len(shifted_steps) != len(raw_steps) or len(x_steps) != len(raw_steps):
                            raise RuntimeError("incomplete clean/shift velocity diagnostics")
                        clean_velocities.append(torch.stack(clean_steps)); shifted_velocities.append(torch.stack(shifted_steps))
                        flow_states.append(torch.stack(x_steps))
                    queue.extend(prefix); traces.append({"replan":replans,**trace}); replans+=1
                model_action=queue.pop(0)
                if not torch.isfinite(model_action).all(): raise RuntimeError("nonfinite action")
                physical=postprocessor(model_action); legal=env_postprocessor({"action":physical})["action"]; observation,_,terminated,_,info=env.step(legal.detach().cpu().numpy()); actions.append(model_action[0].detach().float().cpu()); success=bool(vector_info_value(info,"is_success"))
                if bool(terminated[0]) or success: break
        finally: env.close()
        action_array=torch.stack(actions); diffs=torch.linalg.vector_norm(action_array[1:]-action_array[:-1],dim=1); total=float(diffs.sum()) if len(action_array)>1 else 0.; first30=float(diffs[:29].sum()) if len(action_array)>1 else 0.; skip_steps=[step["skipped_step_due_to_boundary_bool"] for trace in traces for step in trace.get("step_traces",[]) if "skipped_step_due_to_boundary_bool" in step]
        active_steps=[step["active_earlier_self_bool"] for trace in traces for step in trace.get("step_traces",[]) if "active_earlier_self_bool" in step]
        timestep_active=[step["active_timestep_shift_bool"] for trace in traces for step in trace.get("step_traces",[]) if "active_timestep_shift_bool" in step]
        correction_metadata={"correction_vector_path":None,"correction_vector_sha256":None,"correction_vector_shape":None}
        if record_vectors and spec["arm"] != "A_vanilla":
            if len(correction_vectors) != replans: raise RuntimeError("missing correction vector replan")
            raw_tensor=torch.stack(correction_vectors); applied_tensor=torch.stack(applied_correction_vectors)
            correction_output=artifact/"corrections"/f"{spec['episode_id']}.pt"; temporary=correction_output.with_suffix(".pt.tmp")
            if correction_output.exists() or temporary.exists(): raise FileExistsError(correction_output)
            if len(clean_velocities) != replans or len(shifted_velocities) != replans or len(flow_states) != replans:
                raise RuntimeError("missing clean/shift velocity diagnostics")
            torch.save({"raw_clean_minus_branch":raw_tensor,"actual_applied":applied_tensor,"clean_velocity":torch.stack(clean_velocities),"shifted_velocity":torch.stack(shifted_velocities),"x_tau":torch.stack(flow_states),"arm":spec["arm"],"task_id":spec["task_id"],"init_state_id":spec["init_state_id"]},temporary); temporary.rename(correction_output)
            correction_metadata={"correction_vector_path":str(correction_output.relative_to(artifact)),"correction_vector_sha256":file_sha256(correction_output),"correction_vector_shape":list(raw_tensor.shape)}
        record={**spec,**correction_metadata,"status":"complete","success":success,"terminal_failure_or_timeout":not success,"control_steps":len(actions),"replans":replans,"initial_sim_state_sha256":initial_state_hash,"initial_prepared_input_sha256":initial_prepared_hash,"first_noise_sha256_by_replan":first_noise_hashes,"action_total_variation":total,"action_total_variation_first_30_steps":first30,"chunk_discontinuity":float(np.mean(boundaries)) if boundaries else 0.,"median_replan_latency_seconds":float(np.median(latencies)) if latencies else 0.,"fraction_of_flow_steps_with_guidance_skipped":float(np.mean(skip_steps)) if skip_steps else 0.,"fraction_of_flow_steps_with_relative_self_active":float(np.mean(active_steps)) if active_steps else 0.,"fraction_of_flow_steps_with_timestep_shift_active":float(np.mean(timestep_active)) if timestep_active else 0.,"all_actions_finite":bool(torch.isfinite(action_array).all()),"replan_traces":traces,"peak_cuda_memory_bytes":int(torch.cuda.max_memory_allocated())}
        if not record["all_actions_finite"] or not all(math.isfinite(record[k]) for k in ("action_total_variation","action_total_variation_first_30_steps","chunk_discontinuity","median_replan_latency_seconds","fraction_of_flow_steps_with_guidance_skipped")): raise RuntimeError("episode numeric integrity failure")
        with output.open("x",encoding="utf-8") as stream: stream.write(json.dumps(record,indent=2,sort_keys=True)+"\n")
        print(json.dumps({"shard":args.shard_index,"ordinal":ordinal,"episode_id":spec["episode_id"],"success":success,"steps":len(actions)}), flush=True)


if __name__ == "__main__": main()
