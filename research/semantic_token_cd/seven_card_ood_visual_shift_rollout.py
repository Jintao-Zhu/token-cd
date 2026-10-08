"""Paired OpenVLA OOD visual-shift rollouts (Card 7)."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from research.semantic_token_cd import prompt_attn_l11_budget_causal_rollout as shared
from research.semantic_token_cd.distractor_rollout import array_sha256, restore_snapshot, snapshot_sha
from research.semantic_token_cd.distractor_policy import AuditedVanillaInference
from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.spatial_grid_rollout import run_episode
from research.semantic_token_cd.rollout_pilot import wrapped_observation
from research.semantic_token_cd.ood_visual_shift_utils import (
    apply_arena_texture_pixels, apply_sim_visual_shift, apply_static_distractor_pixels,
    create_arena_texture,
    reset_sim_visual_shift,
)

TASKS = tuple(shared.RT_DEPTH)
SHIFTS = ("lighting", "texture", "distractor")
ARMS = ("vanilla", "l11_matched")
PROTOCOL = "SEVEN_CARD_C7_OPENVLA_CONTROLLED_VISUAL_SHIFTS_V1"


def sha(data: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--snapshot-artifact", type=Path, default=Path("artifacts/vanilla_recon_shr_canonical_0_299_v2"))
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--shifts", default=",".join(SHIFTS))
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()
    shifts = tuple(x.strip() for x in args.shifts.split(",") if x.strip())
    if not shifts or any(s not in SHIFTS for s in shifts):
        raise ValueError(f"unsupported visual shifts: {shifts}")
    seeds = [100] if args.preflight else shared.parse_seeds(args.seeds)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import torch
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.distractor_rollout import PCD_SOURCE, get_image_from_maniskill2_obs_dict
    from research.semantic_token_cd.prompt_attn_shr_rollout import atomic_json, write_arrays

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Imported SIMPLER/OpenVLA modules may already have initialized a
        # parallel backend; intra-op threads remain capped above.
        pass
    torch.manual_seed(0); np.random.seed(0); torch.cuda.manual_seed_all(0)
    root = args.artifact.resolve(); root.mkdir(parents=True, exist_ok=True)
    snapshot_root = args.snapshot_artifact.resolve()
    texture_path = create_arena_texture(root / "assets" / "arena_ood_texture.png")
    config = {
        "protocol_id": PROTOCOL, "model": "OpenVLA Fractal SIMPLER checkpoint",
        "tasks": list(TASKS), "seeds": list(range(100, 200)), "arms": list(ARMS),
        "shifts": list(SHIFTS),
        "rendering": {t: {"shader_dir": "rt", "rt_max_path_depth": shared.RT_DEPTH[t],
                           "rt_samples_per_pixel": 32, "rt_use_denoiser": True} for t in TASKS},
        "lighting": "SAPIEN scene ambient set to RGB [0.55,0.55,0.55] after each snapshot restore",
        "texture": "deterministic 256x256 procedural held-out grain blended 0.55 on arena-only pixels identified by Segmentation actor id; arena mesh material has no effective UVs on this renderer",
        "distractor": "render-only 24px checker patch anchored at the highest-arena-fraction corner from baseline segmentation; each frame composites only pixels still segmented as arena, preserving any foreground robot/object that later passes over the fixed screen-space patch",
        "texture_asset": str(texture_path),
        "pairing": "for each task/seed/shift require exact simulator-state and shifted-RGB hashes between Vanilla and L11-Matched; seed snapshot must match canonical RT task reference state",
        "selector": "clean positive post-softmax zero-based L11 prompt-to-visual attention; canonical dynamic Matched budget",
        "counterfactual": "canonical beta=0 harmonic, lambda=0.5, clean-prefix contrastive guidance, dimensions 0-5, clean gripper",
    }
    lock = root / "CONFIG_LOCK.json"
    if lock.exists() and json.loads(lock.read_text()) != config:
        raise RuntimeError("Card 7 CONFIG_LOCK mismatch")
    write_json(lock, config)

    env, environment_id = shared.make_environment(args.task)
    model_config = get_policy_config("openvla", str(PCD_SOURCE / "pretrained/openvla-7b"), args.task, {}, False)
    base = OpenVLAInference(**model_config)
    vanilla = copy.copy(base); vanilla.__class__ = AuditedVanillaInference
    vanilla._episode_trace, vanilla._episode_logits = [], []
    guided = copy.copy(base); guided.__class__ = PromptAttentionSHRInference
    _init_common(guided, 0.5)
    guided.beta = 0.0; guided.selector_mode = "prompt_attention"
    guided.task_index = TASK_INDEX[args.task]; guided.attention_layers = (11,)
    guided.selection_count = None; guided.selection_top_p = None
    guided.selection_budget_schedule = None; guided.selection_budget_scale = 1.0
    guided.selection_budget_source = "matched"; guided.selection_budget_label = "l11_matched"
    guided.counterfactual_constructor = "harmonic"; guided.guidance_execution = "contrast"
    guided.save_prompt_attention = False
    policies = {"vanilla": vanilla, "l11_matched": guided}
    task_dir = root / "episodes" / args.task
    snapshot_dir = snapshot_root / "snapshots" / args.task
    texture = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(texture_path).convert("RGB"))

    try:
        for seed in seeds:
            snapshot_path = snapshot_dir / f"seed_{seed:03d}.pkl"
            if not snapshot_path.is_file():
                raise FileNotFoundError(snapshot_path)
            snapshot = pickle.loads(snapshot_path.read_bytes())
            ref = shared.load_task_reference(snapshot_root, args.task, seed)
            canonical_sha = snapshot_sha(snapshot)
            if ref["canonical_snapshot_sha256"] != canonical_sha:
                raise RuntimeError(f"canonical snapshot mismatch: {args.task}:{seed}")
            expected_state = ref["initial_state_sha256"]
            for shift in shifts:
                paired_hashes = []
                shift_audits = []
                for arm, policy in policies.items():
                    out = task_dir / shift / arm
                    summary_path = out / f"episode_{seed:03d}_summary.json"
                    arrays_path = out / f"episode_{seed:03d}_arrays.npz"
                    if summary_path.exists() and arrays_path.exists():
                        old = json.loads(summary_path.read_text())
                        if old.get("protocol_id") != PROTOCOL:
                            raise RuntimeError(f"incompatible existing output {summary_path}")
                        paired_hashes.append((old["initial_state_sha256"], old["initial_rgb_sha256"]))
                        continue
                    obs, state_hash, _ = restore_snapshot(env, seed, snapshot)
                    if state_hash != expected_state:
                        raise RuntimeError(f"snapshot state mismatch: {args.task}:{seed}")
                    reset_sim_visual_shift(env)
                    obs = wrapped_observation(env)
                    _base_image = get_image_from_maniskill2_obs_dict(env, obs)
                    base_rgb_hash = hashlib.sha256(np.ascontiguousarray(_base_image).tobytes()).hexdigest()
                    audit = apply_sim_visual_shift(
                        env, shift, texture_path if shift == "texture" else None,
                        observation=obs if shift == "distractor" else None,
                    )
                    if shift in {"lighting", "distractor"}:
                        obs = wrapped_observation(env)
                    post_shift_state_hash = array_sha256(np.asarray(env.unwrapped.get_state()))
                    if post_shift_state_hash != state_hash:
                        raise RuntimeError(f"visual shift changed simulator state: {args.task}:{seed}:{shift}")
                    instruction = env.unwrapped.get_language_instruction()
                    if instruction != ref["instruction"]:
                        raise RuntimeError(f"instruction mismatch: {args.task}:{seed}")
                    arena = next((a for a in env.unwrapped._scene.get_all_actors()
                                  if getattr(a, "name", "") == "arena"), None)
                    arena_id = int(arena.get_id()) if arena is not None else -1

                    def transform(image, current_obs):
                        segmentation = np.asarray(current_obs["image"]["overhead_camera"]["Segmentation"])
                        if shift == "texture":
                            return apply_arena_texture_pixels(image, segmentation, texture, arena_id)
                        if shift == "distractor":
                            return apply_static_distractor_pixels(
                                image, segmentation, arena_id,
                                tuple(audit["distractor_camera_pixel_target"]),
                                size=audit["distractor_patch_size_px"][0],
                            )
                        return image

                    initial_image = get_image_from_maniskill2_obs_dict(env, obs)
                    shifted_initial = transform(initial_image, obs)
                    rgb_hash = hashlib.sha256(np.ascontiguousarray(shifted_initial).tobytes()).hexdigest()
                    if rgb_hash == base_rgb_hash:
                        raise RuntimeError(f"visual shift did not change RGB: {args.task}:{seed}:{shift}")
                    policy.reset(instruction, seed=seed)
                    if hasattr(policy, "_episode_trace"):
                        policy._episode_trace, policy._episode_logits = [], []
                    started = time.monotonic()
                    result, steps, failure, actions, jerk = run_episode(
                        env, policy, instruction, obs,
                        observation_transform=transform,
                        max_steps=4 if args.preflight else None,
                    )
                    trace = list(getattr(policy, "_episode_trace", []))
                    if arm == "l11_matched" and not trace:
                        raise RuntimeError("L11-Matched produced an empty selector trace")
                    if arm == "l11_matched":
                        audit["policy_trace_technical_pass"] = all(
                            x.get("feature_equal") is True
                            and x.get("non_target_bit_identical") is True
                            and x.get("reconstruction_finite") is True
                            and x.get("attention_layers") == [11]
                            for x in trace
                        )
                        if not audit["policy_trace_technical_pass"]:
                            raise RuntimeError(f"L11-Matched technical audit failed: {args.task}:{seed}:{shift}")
                    out.mkdir(parents=True, exist_ok=True)
                    logits = getattr(policy, "_episode_logits", [])
                    write_arrays(arrays_path, logits, actions)
                    payload = {
                        "protocol_id": PROTOCOL, "task": args.task, "seed": seed,
                        "shift": shift, "arm": arm, "success": bool(result["success"]),
                        "result": result, "failure_reason": failure, "steps": steps,
                        "runtime_seconds": time.monotonic() - started,
                        "worker_id": args.worker_id, "gpu_id": args.gpu,
                        "environment_id": environment_id,
                        "canonical_snapshot_sha256": canonical_sha,
                        "initial_state_sha256": state_hash,
                        "initial_rgb_sha256": rgb_hash,
                        "base_initial_rgb_sha256": base_rgb_hash,
                        "instruction": instruction, "renderer": {"backend": "SAPIEN RT",
                            "depth": shared.RT_DEPTH[args.task], "spp": 32, "denoiser": True},
                        "shift_audit": audit, "attention_layers": [11] if arm == "l11_matched" else [],
                        "selector_trace": trace, "action_jitter_index": jerk,
                        "arrays_file": arrays_path.name,
                    }
                    atomic_json(summary_path, payload)
                    paired_hashes.append((state_hash, rgb_hash)); shift_audits.append(audit)
                    print(json.dumps({"task": args.task, "seed": seed, "shift": shift,
                                      "arm": arm, "success": payload["success"],
                                      "steps": steps, "runtime_seconds": round(payload["runtime_seconds"], 1)}, sort_keys=True),
                          flush=True)
                if len({state for state, _ in paired_hashes}) != 1 or len({rgb for _, rgb in paired_hashes}) != 1:
                    raise RuntimeError(f"paired shifted state/RGB mismatch: {args.task}:{seed}:{shift}: {paired_hashes}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
