#!/usr/bin/env python3
"""Same-host, same-OpenVLA, same-observation PCD vs L11-Matched latency audit.

The end-to-end boundary is: RGB+segmentation observation available -> decoded,
postprocessed action returned. Simulator rendering/step and model initialization
are excluded. PCD uses the repository default GT segmentation + LaMa inpainting.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import random
import statistics
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
TOKEN_CD = WORKSPACE / "token-cd"
PCD_SOURCE = WORKSPACE / "pcd_openvla_simpler_box_31b027e" / "source"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--task", default="google_robot_pick_coke_can")
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    sys.path[:0] = [str(TOKEN_CD / "task1/shim_site"), str(TOKEN_CD), str(PCD_SOURCE)]

    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    from properties import get_policy_config, get_contrast_image_generator_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from contrast_policies.openvla_contrast import OpenVLAContrastInference
    from contrast_utils import get_contrast_image_generator
    from research.semantic_token_cd.prompt_attn_l11_budget_causal_rollout import make_environment
    from research.semantic_token_cd.distractor_rollout import restore_snapshot
    from research.semantic_token_cd.l11_table2_openvla_rollout import build_policies

    snapshot_path = TOKEN_CD / "artifacts/vanilla_recon_shr_canonical_0_299_v2" / "snapshots" / args.task / f"seed_{args.seed:03d}.pkl"
    snapshot = pickle.loads(snapshot_path.read_bytes())
    env, env_id = make_environment(args.task)
    obs, state_hash, rgb_hash = restore_snapshot(env, args.seed, snapshot)
    image = np.asarray(obs["image"]["overhead_camera"]["rgb"], dtype=np.uint8)
    camera = obs["image"]["overhead_camera"]
    seg = np.asarray(camera["Segmentation"])
    instruction = str(env.unwrapped.get_language_instruction())
    expected = np.load(TOKEN_CD / "artifacts/paper_readiness_audit_v1/latency/canonical_observation.npz", allow_pickle=False)
    if not np.array_equal(image, expected["image"]):
        raise RuntimeError("canonical RGB differs from the existing latency observation")
    if state_hash != str(expected["state_hash"].item()) or rgb_hash != str(expected["rgb_hash"].item()):
        raise RuntimeError("canonical state/RGB hash mismatch")

    # Exact PCD repository defaults: simulator GT mask and LaMa inpainting.
    pcd_gen_cfg = dict(get_contrast_image_generator_config({}))
    pcd_gen_cfg["env"] = env
    # LaMa's upstream Lightning checkpoint contains a serialized module object.
    # PyTorch >=2.6 defaults to weights_only=True, while PCD's vendored loader
    # predates that change. Permit the trusted, hash-verified LaMa checkpoint
    # only; leave all other torch.load calls on their normal safe defaults.
    original_torch_load = torch.load
    lama_checkpoint = str(PCD_SOURCE / "pretrained/big-lama/models/best.ckpt")
    def _load_lama_checkpoint_compat(f, *args, **kwargs):
        if os.path.realpath(os.fspath(f)) == os.path.realpath(lama_checkpoint):
            kwargs.setdefault("weights_only", False)
        return original_torch_load(f, *args, **kwargs)
    torch.load = _load_lama_checkpoint_compat
    pcd_gen = get_contrast_image_generator(pcd_gen_cfg)
    torch.load = original_torch_load
    mask_started = time.perf_counter()
    contrast_image = pcd_gen.generate(obs, instruction)
    torch.cuda.synchronize()
    mask_smoke_ms = (time.perf_counter() - mask_started) * 1000.0
    if contrast_image.shape != image.shape or np.array_equal(contrast_image, image):
        raise RuntimeError("PCD contrast image has invalid shape or no changed pixels")
    pcd_gen.reset()

    model_cfg = get_policy_config(
        "openvla", str(PCD_SOURCE / "pretrained/openvla-7b"), args.task, {}, False
    )
    vanilla = OpenVLAInference(**model_cfg)
    pcd = copy.copy(vanilla)
    pcd.__class__ = OpenVLAContrastInference
    pcd.alpha = 0.2  # latency is independent of this scalar; matches PCD default.
    l11 = build_policies(vanilla, args.task, "6", ("full",))["full"]

    def call_vanilla():
        vanilla.reset(instruction)
        return vanilla.step(image, task_description=instruction)

    def call_pcd_policy_only():
        pcd.reset(instruction)
        return pcd.step(image, contrast_image, task_description=instruction)

    def call_pcd_end_to_end():
        # PCD receives the current RGB+segmentation observation, creates the
        # task-object mask/inpainted image, then runs both policy branches.
        masked = pcd_gen.generate(obs, instruction)
        pcd.reset(instruction)
        return pcd.step(image, masked, task_description=instruction)

    def call_l11_end_to_end():
        l11.reset(instruction)
        return l11.step(image, task_description=instruction)

    calls = {
        "vanilla": call_vanilla,
        "pcd_policy_only_precomputed_mask": call_pcd_policy_only,
        "pcd_end_to_end_gt_mask_lama": call_pcd_end_to_end,
        "l11_matched_end_to_end": call_l11_end_to_end,
    }

    # Warm each path equally, then interleave measured calls to reduce clock/order bias.
    for _ in range(args.warmup):
        for fn in calls.values():
            torch.cuda.synchronize(); fn(); torch.cuda.synchronize()

    values = {name: [] for name in calls}
    order = list(calls)
    for i in range(args.steps):
        random.Random(i).shuffle(order)
        for name in order:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            calls[name]()
            torch.cuda.synchronize()
            values[name].append((time.perf_counter() - t0) * 1000.0)

    def summarize(xs):
        return {
            "n": len(xs),
            "mean_ms": statistics.fmean(xs),
            "median_ms": statistics.median(xs),
            "p90_ms": float(np.percentile(xs, 90)),
            "p95_ms": float(np.percentile(xs, 95)),
            "mean_hz": 1000.0 / statistics.fmean(xs),
        }

    result = {
        "protocol": "same H100, same OpenVLA checkpoint, same canonical RT-depth0 RGB+seg observation; RGB+seg ready -> decoded/postprocessed action; CUDA synchronized wall time",
        "excluded": ["model loading", "simulator construction/reset", "rendering", "environment step", "video/log writing"],
        "pcd_mask": {"generator_defaults": {k: v for k, v in pcd_gen_cfg.items() if k != "env"}, "one_smoke_generation_ms": mask_smoke_ms, "changed_pixel_fraction": float(np.mean(np.any(contrast_image != image, axis=-1)))},
        "observation": {"task": args.task, "seed": args.seed, "environment_id": env_id, "instruction": instruction, "rgb_shape": list(image.shape), "seg_shape": list(seg.shape), "state_sha256": state_hash, "rgb_sha256": rgb_hash},
        "checkpoint": str(PCD_SOURCE / "pretrained/openvla-7b"),
        "warmup_per_path": args.warmup,
        "measured_calls_per_path": args.steps,
        "timing_boundary": "before policy.reset + call; after returned action and torch.cuda.synchronize",
        "latency": {name: summarize(xs) for name, xs in values.items()},
        "memory_peak_gib": {
            "note": "Measured in one shared process; LaMa and OpenVLA stay resident, so these are shared-process peaks and not clean per-method incremental VRAM attribution.",
            "allocated": torch.cuda.max_memory_allocated() / 1024**3,
            "reserved": torch.cuda.max_memory_reserved() / 1024**3,
        },
    }
    output = TOKEN_CD / "artifacts/strict_pcd_l11_latency_v1"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"strict_latency_gpu{args.gpu}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    env.close()
    print(json.dumps({"output": str(path), "latency": result["latency"], "mask": result["pcd_mask"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
