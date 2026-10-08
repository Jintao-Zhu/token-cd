"""Offline Pi0 flow hidden-guidance proof of concept (no closed-loop episodes).

Uses canonical SIMPLER initial snapshots, the existing L10-Matched harmonic
visual counterfactual, common initial action noise, and identical flow states
for positive/negative branches. Hidden guidance is injected into the action
expert after a selected layer; the remaining action-expert layers and original
action decoder produce the velocity used by the unchanged Euler sampler.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

ROOT = Path("/home/leju-suzhou/zjt_ws")
REPO = ROOT / "token-cd"
OPENPI = ROOT / "open-pi-zero"
ARTIFACT = REPO / "artifacts/pi0_flow_hidden_guidance_poc_v1"
SNAPSHOTS = REPO / "artifacts/vanilla_recon_shr_canonical_0_299_v2/snapshots"
TASK = "google_robot_pick_coke_can"
SEEDS = (0, 1, 2, 3)
HIDDEN_CUTS = (8, 17)  # zero-based action-expert blocks: middle and final
LAMBDAS = (0.25, 0.5)
LAYER_SELECTOR = 10
PROTOCOL = "PI0_FLOW_HIDDEN_GUIDANCE_OFFLINE_POC_V1"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def run_action_expert(model, embeddings, time_cond, action_mask, action_positions,
                      caches, *, start_layer=0, stop_layer=None):
    """Run action-mixture blocks over a supplied hidden tensor."""
    import torch
    from src.model.vla.joint_model import forward_mixture_layers

    stop_layer = model.joint_model.num_hidden_layers - 1 if stop_layer is None else stop_layer
    state = {"action": embeddings}
    for layer_idx in range(start_layer, stop_layer + 1):
        state = forward_mixture_layers(
            model.joint_model.mixtures,
            action_mask,
            {"action": action_positions},
            state,
            layer_idx=layer_idx,
            time_cond=time_cond,
            kv_caches=caches,
            cache_mode="append_non_active",
        )
    return state["action"]


def build_prefix_caches(model, prefix, proprio, masks, inputs, count=1):
    import torch
    from src.model.kv_cache import KVCache

    causal, vlm_pos, prop_pos = masks
    vlm_prop, _ = model.split_full_mask_into_submasks(causal)
    with torch.inference_mode():
        _, caches = model.joint_model(
            attention_mask=vlm_prop,
            position_ids_all={"vlm": vlm_pos, "proprio": prop_pos},
            embeds_all={"vlm": prefix.clone(), "proprio": proprio.clone()},
            kv_caches=model.joint_model.build_mixture_caches(),
            return_caches=True,
        )
    expanded = {}
    for name, old in caches.items():
        cache = KVCache()
        cache.key_cache = [v.expand(count, *v.shape[1:]) for v in old.key_cache]
        cache.value_cache = [v.expand(count, *v.shape[1:]) for v in old.value_cache]
        expanded[name] = cache
    _, action_mask = model.split_full_mask_into_submasks(causal)
    _, _, _, action_positions = model.build_causal_mask_and_position_ids(
        inputs["attention_mask"], dtype=inputs["pixel_values"].dtype
    )
    return expanded, action_mask, action_positions.to(prefix.device)


def branch_hidden(model, action, t, action_mask, action_positions, caches, cuts):
    """Return pre-final-norm hidden after requested action-expert blocks."""
    import torch
    from src.model.vla.joint_model import forward_mixture_layers

    time_cond = model.time_embedding(t)
    if model.action_expert_adaptive_mode:
        embeddings = model.action_encoder(action)
    else:
        embeddings = model.action_encoder(action, time_cond)
    state = {"action": embeddings}
    captured = {}
    for layer_idx in range(model.joint_model.num_hidden_layers):
        state = forward_mixture_layers(
            model.joint_model.mixtures,
            action_mask,
            {"action": action_positions},
            state,
            layer_idx=layer_idx,
            time_cond=time_cond,
            kv_caches=caches,
            cache_mode="append_non_active",
        )
        if layer_idx in cuts:
            captured[layer_idx] = state["action"].clone()
    final_hidden = state["action"]
    return captured, final_hidden, time_cond


def guided_velocity_from_cut(model, pos_hidden, neg_hidden, time_cond, cut,
                             lambd, action_mask, action_positions, pos_cache):
    import torch
    hidden = pos_hidden + float(lambd) * (pos_hidden - neg_hidden)
    if cut + 1 < model.joint_model.num_hidden_layers:
        hidden = run_action_expert(
            model, hidden, time_cond, action_mask, action_positions, pos_cache,
            start_layer=cut + 1,
        )
    hidden = model.joint_model.mixtures["action"].forward_norm(hidden, time_cond)
    return model.action_decoder(hidden)


def clean_velocity(model, final_hidden, time_cond):
    hidden = model.joint_model.mixtures["action"].forward_norm(final_hidden, time_cond)
    return model.action_decoder(hidden)


def integrate_sample(model, inputs, positive_cache, negative_cache, action_mask,
                     action_positions, cuts, noise, lambd):
    import torch

    x_v = noise.clone()
    x_clean = noise.clone()
    x_h = {cut: noise.clone() for cut in cuts}
    dt = 1.0 / model.num_inference_steps
    t0 = torch.zeros(noise.shape[0], device=noise.device, dtype=noise.dtype)
    per_step = []
    for flow_step in range(model.num_inference_steps):
        _, h_pos_final, time_pos = branch_hidden(
            model, x_v, t0, action_mask, action_positions, positive_cache, ()
        )
        _, h_neg_final, time_neg = branch_hidden(
            model, x_v, t0, action_mask, action_positions, negative_cache, ()
        )
        v_pos = clean_velocity(model, h_pos_final, time_pos)
        v_neg = clean_velocity(model, h_neg_final, time_neg)
        x_clean = x_clean + dt * v_pos
        v_guided = v_pos.clone()
        v_guided[..., :-1] += float(lambd) * (v_pos[..., :-1] - v_neg[..., :-1])
        x_v = x_v + dt * v_guided
        row = {
            "flow_step": flow_step,
            "t": float(t0[0].item()),
            "velocity_guidance_delta_norm": float(torch.linalg.vector_norm((v_guided-v_pos)[..., :-1]).item()),
            "negative_positive_velocity_cosine": float(torch.nn.functional.cosine_similarity(
                (v_pos-v_neg)[..., :-1].float().reshape(1, -1),
                v_pos[..., :-1].float().reshape(1, -1), dim=-1).item()),
            "cuts": {},
        }
        for cut in cuts:
            # Each hidden-guidance arm follows its own flow trajectory. At a
            # given Euler step, its positive and negative branches see the
            # exact same current noisy action state.
            h_pos_captured, h_pos_final_arm, time_pos_arm = branch_hidden(
                model, x_h[cut], t0, action_mask, action_positions, positive_cache, cuts
            )
            h_neg_captured, h_neg_final_arm, time_neg_arm = branch_hidden(
                model, x_h[cut], t0, action_mask, action_positions, negative_cache, cuts
            )
            hidden_pos = h_pos_captured[cut]
            hidden_neg = h_neg_captured[cut]
            hidden_guided_norm = float(torch.linalg.vector_norm(
                hidden_pos + float(lambd) * (hidden_pos-hidden_neg)
            ).item())
            hidden_delta_norm = float(torch.linalg.vector_norm(hidden_pos-hidden_neg).item())
            v_h = guided_velocity_from_cut(
                model, hidden_pos, hidden_neg, time_pos_arm, cut, lambd,
                action_mask, action_positions, positive_cache,
            )
            v_pos_arm = clean_velocity(model, h_pos_final_arm, time_pos_arm)
            v_h[..., -1] = v_pos_arm[..., -1]
            x_h[cut] = x_h[cut] + dt * v_h
            row["cuts"][str(cut)] = {
                "hidden_delta_over_positive_norm": hidden_delta_norm / (float(torch.linalg.vector_norm(hidden_pos).item()) + 1e-8),
                "guided_hidden_norm_over_positive": hidden_guided_norm / (float(torch.linalg.vector_norm(hidden_pos).item()) + 1e-8),
                "guided_velocity_delta_norm": float(torch.linalg.vector_norm((v_h-v_pos_arm)[..., :-1]).item()),
                "guided_velocity_relative_delta": float(torch.linalg.vector_norm((v_h-v_pos_arm)[..., :-1]).item() / (torch.linalg.vector_norm(v_pos_arm[..., :-1]).item() + 1e-8)),
                "gripper_velocity_unchanged": bool(torch.equal(v_h[..., -1], v_pos_arm[..., -1])),
            }
        per_step.append(row)
        t0 = t0 + dt
    if model.final_action_clip_value is not None:
        x_v = torch.clamp(x_v, -model.final_action_clip_value, model.final_action_clip_value)
        x_clean = torch.clamp(x_clean, -model.final_action_clip_value, model.final_action_clip_value)
        for cut in cuts:
            x_h[cut] = torch.clamp(x_h[cut], -model.final_action_clip_value, model.final_action_clip_value)
    results = {"vanilla": x_clean, "velocity_cd": x_v}
    results.update({f"hidden_cd_layer_{cut}": x for cut, x in x_h.items()})
    return results, per_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--seeds", default="0-3")
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--cuts", default=",".join(map(str, HIDDEN_CUTS)),
                        help="comma-separated zero-based action-expert block indices")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise RuntimeError(f"launch with CUDA_VISIBLE_DEVICES={args.gpu}")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "checkpoints/pi0"))
    sys.path[:0] = [str(OPENPI), str(REPO / "task1/shim_site"), str(REPO)]
    os.chdir(OPENPI)

    import numpy as np
    import torch
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    from research.semantic_token_cd.pi0_fractal_simpler_l2_pcd_rollout import (
        make_env, restore_snapshot,
    )
    from research.semantic_token_cd.pi0_l10_pcd_remaining5_rollout import load_model
    from research.semantic_token_cd.pi0_fractal_simpler_l2_pcd_rollout import entity_budget_and_attention
    from research.semantic_token_cd.pi0_intact_token_pcd import harmonic_reconstruct

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:0")
    artifact = args.artifact.resolve()
    cuts = tuple(sorted(set(int(x) for x in args.cuts.split(",") if x.strip())))
    if not cuts or cuts[0] < 0 or cuts[-1] >= 18:
        raise ValueError(f"hidden cuts must be unique indices in 0..17, got {cuts}")
    artifact.mkdir(parents=True, exist_ok=True)
    task = TASK
    env, env_id = make_env(task)
    cfg, model, adapter, checkpoint = load_model(task, device)
    model.eval()
    rows = []
    try:
        for seed_text in args.seeds.split(","):
            seed_text = seed_text.strip()
            if not seed_text:
                continue
            if "-" in seed_text:
                lo, hi = map(int, seed_text.split("-", 1))
                seeds = range(lo, hi + 1)
            else:
                seeds = [int(seed_text)]
            for seed in seeds:
                snapshot_path = SNAPSHOTS / task / f"seed_{seed:03d}.pkl"
                with snapshot_path.open("rb") as f:
                    snapshot = pickle.load(f)
                obs, state_hash, rgb_hash = restore_snapshot(
                    env, seed, snapshot, get_image_from_maniskill2_obs_dict
                )
                instruction = env.unwrapped.get_language_instruction()
                inputs = adapter.preprocess(env, obs, instruction)
                inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
                prefix, image_positions, proprio, masks, matched, ids, attention = entity_budget_and_attention(
                    model, adapter, inputs, instruction, LAYER_SELECTOR
                )
                features = prefix[0, image_positions].detach().float().cpu().numpy()
                reconstructed = harmonic_reconstruct(features, ids)
                negative_prefix = prefix.clone()
                negative_prefix[0, image_positions[ids]] = torch.as_tensor(
                    reconstructed, device=prefix.device, dtype=prefix.dtype
                )
                positive_cache, action_mask, action_positions = build_prefix_caches(
                    model, prefix, proprio, masks, inputs
                )
                negative_cache, _, _ = build_prefix_caches(
                    model, negative_prefix, proprio, masks, inputs
                )
                generator = torch.Generator(device=device).manual_seed(seed * 1009)
                noise = torch.randn(
                    (1, model.horizon_steps, model.action_dim), generator=generator,
                    device=device, dtype=inputs["pixel_values"].dtype,
                )
                sample_row = {
                    "protocol_id": PROTOCOL,
                    "task": task,
                    "environment_id": env_id,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "renderer": {"shader": "rt", "depth": 8, "spp": 32, "denoiser": True},
                    "selector": "L10 prompt-attention ranking with existing Matched dynamic m and harmonic negative",
                    "matched_m": len(matched),
                    "selected_ids_sha256": __import__("hashlib").sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest(),
                    "attention_sha256": __import__("hashlib").sha256(np.ascontiguousarray(attention).tobytes()).hexdigest(),
                    "initial_state_sha256": state_hash,
                    "initial_rgb_sha256": rgb_hash,
                    "action_expert_layers": model.joint_model.num_hidden_layers,
                    "hidden_cut_indices_0based": list(cuts),
                    "initial_noise_sha256": __import__("hashlib").sha256(noise.detach().float().cpu().numpy().tobytes()).hexdigest(),
                    "lambdas": {},
                }
                for lambd in LAMBDAS:
                    with torch.inference_mode():
                        outputs, per_step = integrate_sample(
                            model, inputs, positive_cache, negative_cache,
                            action_mask, action_positions, cuts, noise, lambd,
                        )
                    base = outputs["velocity_cd"]
                    sample_row["lambdas"][str(lambd)] = {
                        "flow_steps": per_step,
                        "arms": {},
                    }
                    for arm, value in outputs.items():
                        delta = (value - base).detach().float()
                        clean_delta = (value - outputs["vanilla"]).detach().float()
                        sample_row["lambdas"][str(lambd)]["arms"][arm] = {
                            "action_chunk_l2_vs_velocity_cd": float(torch.linalg.vector_norm(delta).item()),
                            "action_chunk_mean_abs_vs_velocity_cd": float(delta.abs().mean().item()),
                            "action_chunk_l2_vs_vanilla": float(torch.linalg.vector_norm(clean_delta).item()),
                            "action_chunk_mean_abs_vs_vanilla": float(clean_delta.abs().mean().item()),
                            "finite": bool(torch.isfinite(value).all()),
                            "action_chunk": value.detach().float().cpu().numpy().tolist(),
                        }
                    sample_row["lambdas"][str(lambd)]["arms"]["velocity_cd"]["action_chunk_l2_vs_velocity_cd"] = 0.0
                    sample_row["lambdas"][str(lambd)]["arms"]["velocity_cd"]["action_chunk_mean_abs_vs_velocity_cd"] = 0.0
                    sample_row["lambdas"][str(lambd)]["arms"]["vanilla"]["action_chunk_l2_vs_vanilla"] = 0.0
                    sample_row["lambdas"][str(lambd)]["arms"]["vanilla"]["action_chunk_mean_abs_vs_vanilla"] = 0.0
                rows.append(sample_row)
                atomic_json(artifact / "samples" / f"{task}_seed_{seed:03d}.json", sample_row)
                print(json.dumps({"event": "sample_done", "task": task, "seed": seed,
                                  "m": len(matched), "lambda_values": list(LAMBDAS),
                                  "gpu": args.gpu}), flush=True)
        manifest = {
            "protocol_id": PROTOCOL,
            "scope": "offline inference only; no closed-loop episodes",
            "model": "Pi0 Fractal-Beta SIMPLER checkpoint",
            "renderer": "RT depth 8, 32 spp, denoiser; same as the existing Google Coke PCD environment helper",
            "selector": "existing L10 prompt-attention + Matched budget + harmonic reconstruction",
            "hidden_guidance": "h*=h_pos+lambda*(h_pos-h_neg) at action-expert block output; remaining blocks and original decoder use positive VLM/proprio KV cache; gripper velocity kept positive",
            "baseline": "existing shared-noise flow velocity CD on first six action dimensions",
            "seeds": list(range(int(args.seeds.split("-")[0]), int(args.seeds.split("-")[-1]) + 1)) if "-" in args.seeds else [int(args.seeds)],
            "hidden_cut_indices_0based": list(cuts),
            "lambdas": list(LAMBDAS),
            "num_inference_steps": model.num_inference_steps,
            "checkpoint": str(checkpoint),
        }
        atomic_json(artifact / "CONFIG_LOCK.json", manifest)
        atomic_json(artifact / "SUMMARY.json", {"manifest": manifest, "samples": rows})
    finally:
        env.close()


if __name__ == "__main__":
    main()
