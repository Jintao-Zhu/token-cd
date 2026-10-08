"""Bridge-task Pi0 hidden-guidance POC on initial and partial-rollout states.

This captures at most 40 control steps from one Vanilla trajectory, then runs
paired offline flow comparisons at steps 0, 20, and 40. It does not score or
complete an episode.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

ROOT = Path("/home/leju-suzhou/zjt_ws")
REPO = ROOT / "token-cd"
OPENPI = ROOT / "open-pi-zero"
ARTIFACT = REPO / "artifacts/pi0_flow_hidden_guidance_poc_v1"
TASK = "widowx_carrot_on_plate"
SEED = 0
PHASE_STEPS = (0, 20, 40)
SCAN_CUTS = (2, 5, 8, 11, 14, 17)
OUTPUT_SUBDIR = "bridge_layer_scan_samples"


def sha256_array(value):
    import numpy as np
    value = np.asarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def prepared_inputs(model, adapter, env, obs, instruction, device):
    import torch
    x = adapter.preprocess(env, obs, instruction)
    dtype = next(model.parameters()).dtype
    causal, vlm_pos, proprio_pos, action_pos = model.build_causal_mask_and_position_ids(
        x["attention_mask"], dtype=dtype
    )
    cond_mask, action_mask = model.split_full_mask_into_submasks(causal)
    ready = {
        "input_ids": x["input_ids"],
        "pixel_values": x["pixel_values"].to(dtype),
        "image_text_proprio_mask": cond_mask,
        "action_mask": action_mask,
        "vlm_position_ids": vlm_pos,
        "proprio_position_ids": proprio_pos,
        "action_position_ids": action_pos,
        "proprios": x["proprios"].to(dtype),
    }
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in ready.items()}


def evaluate_state(env, obs, model, adapter, instruction, device, seed, phase):
    import numpy as np
    import torch
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    from research.semantic_token_cd.pi0_fractal_simpler_l2_pcd_rollout import entity_budget_and_attention
    from research.semantic_token_cd.pi0_intact_token_pcd import harmonic_reconstruct
    from research.semantic_token_cd.pi0_flow_hidden_guidance_poc import (
        LAMBDAS, LAYER_SELECTOR, PROTOCOL,
        build_prefix_caches, integrate_sample,
    )

    x = adapter.preprocess(env, obs, instruction)
    x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in x.items()}
    prefix, image_positions, proprio, masks, matched, ids, attention = entity_budget_and_attention(
        model, adapter, x, instruction, LAYER_SELECTOR
    )
    features = prefix[0, image_positions].detach().float().cpu().numpy()
    reconstructed = harmonic_reconstruct(features, ids)
    negative_prefix = prefix.clone()
    negative_prefix[0, image_positions[ids]] = torch.as_tensor(
        reconstructed, device=prefix.device, dtype=prefix.dtype
    )
    pos_cache, action_mask, action_positions = build_prefix_caches(
        model, prefix, proprio, masks, x
    )
    neg_cache, _, _ = build_prefix_caches(model, negative_prefix, proprio, masks, x)
    generator = torch.Generator(device=device).manual_seed(seed * 1009 + phase)
    noise = torch.randn((1, model.horizon_steps, model.action_dim), generator=generator,
                        device=device, dtype=x["pixel_values"].dtype)
    row = {
        "protocol_id": PROTOCOL,
        "task": TASK,
        "seed": seed,
        "trajectory_source": "partial Vanilla rollout using Bridge checkpoint; not a success evaluation",
        "phase_control_steps": phase,
        "matched_m": len(matched),
        "selected_ids_sha256": hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest(),
        "attention_sha256": hashlib.sha256(np.ascontiguousarray(attention).tobytes()).hexdigest(),
        "state_sha256": sha256_array(env.unwrapped.get_state()),
        "rgb_sha256": sha256_array(get_image_from_maniskill2_obs_dict(env, obs)),
        "lambdas": {},
    }
    for lambd in LAMBDAS:
        with torch.inference_mode():
            outputs, per_step = integrate_sample(
            model, x, pos_cache, neg_cache, action_mask, action_positions,
                SCAN_CUTS, noise, lambd,
            )
        row["lambdas"][str(lambd)] = {
            "flow_steps": per_step,
            "arms": {},
        }
        vanilla = outputs["vanilla"]
        velocity = outputs["velocity_cd"]
        for arm, value in outputs.items():
            clean_delta = (value - vanilla).detach().float()
            velocity_delta = (value - velocity).detach().float()
            row["lambdas"][str(lambd)]["arms"][arm] = {
                "action_chunk": value.detach().float().cpu().numpy().tolist(),
                "action_chunk_l2_vs_vanilla": float(torch.linalg.vector_norm(clean_delta).item()),
                "action_chunk_mean_abs_vs_vanilla": float(clean_delta.abs().mean().item()),
                "action_chunk_l2_vs_velocity_cd": float(torch.linalg.vector_norm(velocity_delta).item()),
                "finite": bool(torch.isfinite(value).all()),
            }
    return row


def main():
    gpu = int(os.environ.get("POC_GPU", "1"))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError(f"launch with CUDA_VISIBLE_DEVICES={gpu}")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "checkpoints/pi0"))
    sys.path[:0] = [str(OPENPI), str(REPO / "task1/shim_site"), str(REPO)]
    os.chdir(OPENPI)
    import numpy as np
    import torch
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    from research.semantic_token_cd.pi0_fractal_simpler_l2_pcd_rollout import restore_snapshot
    from research.semantic_token_cd.pi0_l10_pcd_remaining5_rollout import make_env, load_model
    from research.semantic_token_cd.pi0_shr_policy import Pi0SHRInference
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:0")

    env, env_id = make_env(TASK)
    cfg, model, adapter, checkpoint = load_model(TASK, device)
    policy = Pi0SHRInference(model, adapter.tokenizer)
    snapshot_path = REPO / "artifacts/vanilla_recon_shr_canonical_0_299_v2/snapshots" / TASK / f"seed_{SEED:03d}.pkl"
    with snapshot_path.open("rb") as f:
        snapshot = pickle.load(f)
    obs, initial_state_hash, initial_rgb_hash = restore_snapshot(
        env, SEED, snapshot, get_image_from_maniskill2_obs_dict
    )
    instruction = env.unwrapped.get_language_instruction()
    output_rows = []
    try:
        phase_targets = set(PHASE_STEPS)
        phase = 0
        output_rows.append(evaluate_state(env, obs, model, adapter, instruction, device, SEED, phase))
        steps = 0
        replan = 0
        max_steps = max(PHASE_STEPS)
        while steps < max_steps:
            inputs = prepared_inputs(model, adapter, env, obs, instruction, device)
            action_chunk, _ = policy.infer(
                inputs, instruction, arm="pi0_vanilla", noise_seed=SEED * 1009 + replan
            )
            env_actions = adapter.postprocess(action_chunk[0].float().cpu().numpy())
            for action in env_actions[: int(cfg.act_steps)]:
                obs, _reward, _success, truncated, _info = env.step(np.asarray(action, dtype=np.float64))
                steps += 1
                if steps in phase_targets and steps != 0:
                    instruction = env.unwrapped.get_language_instruction()
                    output_rows.append(evaluate_state(
                        env, obs, model, adapter, instruction, device, SEED, steps
                    ))
                if truncated or steps >= max_steps:
                    break
            instruction = env.unwrapped.get_language_instruction()
            replan += 1
        artifact = ARTIFACT / OUTPUT_SUBDIR
        artifact.mkdir(parents=True, exist_ok=True)
        for row in output_rows:
            row.update({
                "environment_id": env_id,
                "checkpoint": str(checkpoint),
                "renderer": {"shader": "rt", "depth": 8, "spp": 32, "denoiser": True},
                "initial_state_sha256": initial_state_hash,
                "initial_rgb_sha256": initial_rgb_hash,
                "action_expert_layers": model.joint_model.num_hidden_layers,
            })
            path = artifact / f"{TASK}_seed_{SEED:03d}_step_{row['phase_control_steps']:03d}.json"
            path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
        manifest = {
            "scope": "one partial Vanilla trajectory to control step 40 plus offline branch evaluations; not a completed/ scored episode",
            "task": TASK,
            "seed": SEED,
            "checkpoint": str(checkpoint),
            "renderer": {"shader": "rt", "depth": 8, "spp": 32, "denoiser": True},
            "captured_phases": [row["phase_control_steps"] for row in output_rows],
            "hidden_cut_indices_0based": list(SCAN_CUTS),
            "initial_state_sha256": initial_state_hash,
            "initial_rgb_sha256": initial_rgb_hash,
            "flow_guidance": "shared noisy action state per positive/negative pair; same L10/Matched/harmonic counterfactual; hidden injection then positive remaining action-expert layers and original decoder",
        }
        (artifact / "CONFIG_LOCK.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"event": "phase_poc_complete", "task": TASK,
                          "phases": [r["phase_control_steps"] for r in output_rows],
                          "cuts": list(SCAN_CUTS),
                          "gpu": gpu, "checkpoint": str(checkpoint)}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
