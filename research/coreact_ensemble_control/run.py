from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from research.coreact_closed_loop.guidance import GuidanceConfig, sample_coreact_actions, tensor_sha256
from research.coreact_closed_loop.run_pilot import vector_info_value
from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare
from research.coreact_ensemble_control.methods import sample_dual_seed_average


ARMS = {"A_vanilla", "B_toward_top8", "C_dual_seed_chunk_average", "D_random8_toward"}


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def prepared_sha256(batch: dict) -> str:
    values = [*batch["images"], *batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"]]
    return hashlib.sha256("".join(tensor_sha256(value) for value in values).encode("ascii")).hexdigest()


def jaccard(first: list[int], second: list[int] | None) -> float | None:
    if second is None:
        return None
    union = set(first) | set(second)
    return len(set(first) & set(second)) / len(union) if union else 1.0


def summarize_guidance_trace(trace: dict, replan: int, previous: list[int] | None) -> dict:
    steps = trace["step_traces"]
    selected = trace["selected_indices"]
    return {
        "replan": replan,
        "method": "toward_top8" if trace["selection"] == "top" else "toward_random8",
        "clean_minus_masked_l2_norm_pre_clip": [float(step["negative_delta_norm"]) for step in steps],
        "applied_delta_l2_norm_post_clip": [float(step["applied_guidance_norm"]) for step in steps],
        "trust_region_clipping_active_bool": any(float(step["clip_scale"]) < 1.0 - 1e-12 for step in steps),
        "clip_scale_by_flow_step": [float(step["clip_scale"]) for step in steps],
        "selected_token_indices": selected,
        "changed_token_indices": trace["changed_indices"],
        "attention_top8_indices": trace["attention_top_indices"],
        "jaccard_with_previous_replan_selection": jaccard(selected, previous),
        "overlap_with_attention_top8": trace["overlap_with_attention_top"],
        "prefix_sha256": trace["prefix_sha256"],
        "masked_prefix_sha256": trace["negative_prefix_sha256"],
        "protected_tokens_untouched": trace["protected_tokens_untouched"],
        "all_output_finite": trace["all_output_finite"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    gate = json.loads((artifact / "integrity_report.json").read_text())
    if not gate.get("pass") or not (artifact / "status/integrity.pass").exists():
        raise RuntimeError("integrity gate has not passed")
    manifest = [json.loads(line) for line in (artifact / "episode_manifest.jsonl").read_text().splitlines()]
    unknown = {row["arm"] for row in manifest} - ARMS
    if unknown:
        raise RuntimeError(f"unknown arms: {sorted(unknown)}")
    rows = [row for index, row in enumerate(manifest) if index % args.shard_count == args.shard_index]
    config, policy, preprocessor, postprocessor = load_policy_and_processors(workspace)
    means = torch.load(workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt", weights_only=True, map_location="cpu")
    top_config = GuidanceConfig(selection="top", group_count=8, guidance_scale=0.5, trust_region_kappa=0.25, action_dim=7, num_steps=10, direction="toward")
    random_config = GuidanceConfig(selection="random", group_count=8, guidance_scale=0.5, trust_region_kappa=0.25, action_dim=7, num_steps=10, direction="toward")
    for ordinal, spec in enumerate(rows, 1):
        output = artifact / "episodes" / f"{spec['episode_id']}.json"
        if output.exists():
            continue
        env, env_preprocessor, env_postprocessor = make_task_env(spec["suite"], spec["task_id"], config)
        actions: list[torch.Tensor] = []
        queue: list[torch.Tensor] = []
        traces: list[dict] = []
        boundaries: list[float] = []
        latencies: list[float] = []
        first_noise_hashes: list[str] = []
        second_noise_hashes: list[str] = []
        replans = 0
        success = False
        previous_selection = None
        try:
            inner = env.envs[0]
            inner.init_state_id = spec["init_state_id"]
            observation, _ = env.reset(seed=spec["reset_seed"])
            initial_state_hash = array_sha256(np.asarray(inner._env.get_sim_state()))
            initial_prepared_hash = None
            torch.cuda.reset_peak_memory_stats()
            for _control_step in range(280):
                if not queue:
                    batch = prepare(policy, preprocessor, env_preprocessor, observation, spec["language"])
                    if initial_prepared_hash is None:
                        initial_prepared_hash = prepared_sha256(batch)
                    shape = (1, config.chunk_size, config.max_action_dim)
                    generator = torch.Generator(device=batch["state"].device)
                    noise1 = torch.randn(shape, generator=generator.manual_seed(spec["action_noise_seed"] * 1000 + replans), device=batch["state"].device, dtype=batch["state"].dtype)
                    noise2 = torch.randn(shape, generator=generator.manual_seed(spec["second_action_noise_seed"] * 1000 + replans), device=batch["state"].device, dtype=batch["state"].dtype)
                    first_noise_hashes.append(tensor_sha256(noise1))
                    started = time.perf_counter()
                    with torch.inference_mode():
                        if spec["arm"] == "A_vanilla":
                            chunk = policy.model.sample_actions(batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"], noise=noise1)
                            trace = {"replan": replans, "method": "vanilla", "selected_token_indices": [], "changed_token_indices": [], "first_noise_sha256": tensor_sha256(noise1), "all_output_finite": bool(torch.isfinite(chunk).all())}
                        elif spec["arm"] == "B_toward_top8":
                            chunk, raw_trace = sample_coreact_actions(policy.model, batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"], noise1, means["visual_position_mean"], config=top_config, selection_seed=spec["selection_seed"] * 1000 + replans)
                            trace = summarize_guidance_trace(raw_trace, replans, previous_selection)
                        elif spec["arm"] == "C_dual_seed_chunk_average":
                            chunk, raw_trace = sample_dual_seed_average(policy.model, batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"], noise1, noise2, matched_step=True)
                            second_noise_hashes.append(tensor_sha256(noise2))
                            trace = {"replan": replans, **raw_trace, "selected_token_indices": [], "changed_token_indices": [], "jaccard_with_previous_replan_selection": None, "overlap_with_attention_top8": None}
                        elif spec["arm"] == "D_random8_toward":
                            chunk, raw_trace = sample_coreact_actions(policy.model, batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"], noise1, means["visual_position_mean"], config=random_config, selection_seed=spec["selection_seed"] * 1000 + replans)
                            trace = summarize_guidance_trace(raw_trace, replans, previous_selection)
                        else:
                            raise RuntimeError(f"unknown arm {spec['arm']}")
                    torch.cuda.synchronize()
                    latencies.append(time.perf_counter() - started)
                    if not torch.isfinite(chunk).all():
                        raise RuntimeError("nonfinite sampler output")
                    prefix = chunk[:, :10, :7].transpose(0, 1)
                    if actions:
                        boundaries.append(float(torch.linalg.vector_norm(prefix[0, 0].cpu() - actions[-1])))
                    queue.extend(prefix)
                    traces.append(trace)
                    previous_selection = trace["selected_token_indices"] or previous_selection
                    replans += 1
                model_action = queue.pop(0)
                if not torch.isfinite(model_action).all():
                    raise RuntimeError("nonfinite action")
                physical = postprocessor(model_action)
                legal = env_postprocessor({"action": physical})["action"]
                observation, _, terminated, _, info = env.step(legal.detach().cpu().numpy())
                actions.append(model_action[0].detach().float().cpu())
                success = bool(vector_info_value(info, "is_success"))
                if bool(terminated[0]) or success:
                    break
        finally:
            env.close()
        action_array = torch.stack(actions)
        deltas = torch.linalg.vector_norm(action_array[1:] - action_array[:-1], dim=1)
        total_variation = float(deltas.sum()) if len(action_array) > 1 else 0.0
        first_30_variation = float(deltas[:29].sum()) if len(action_array) > 1 else 0.0
        record = {
            **spec,
            "status": "complete",
            "success": success,
            "terminal_failure_or_timeout": not success,
            "control_steps": len(actions),
            "replans": replans,
            "initial_sim_state_sha256": initial_state_hash,
            "initial_prepared_input_sha256": initial_prepared_hash,
            "first_noise_sha256_by_replan": first_noise_hashes,
            "second_noise_sha256_by_replan": second_noise_hashes,
            "action_total_variation": total_variation,
            "action_total_variation_first_30_steps": first_30_variation,
            "chunk_discontinuity": float(np.mean(boundaries)) if boundaries else 0.0,
            "median_replan_latency_seconds": float(np.median(latencies)) if latencies else 0.0,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "all_actions_finite": bool(torch.isfinite(action_array).all()),
            "replan_traces": traces,
        }
        if not record["all_actions_finite"] or not all(math.isfinite(record[key]) for key in ("action_total_variation", "action_total_variation_first_30_steps", "chunk_discontinuity", "median_replan_latency_seconds")):
            raise RuntimeError("episode numeric integrity failure")
        with output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"shard": args.shard_index, "ordinal": ordinal, "episode_id": spec["episode_id"], "success": success, "steps": len(actions)}), flush=True)


if __name__ == "__main__":
    main()
