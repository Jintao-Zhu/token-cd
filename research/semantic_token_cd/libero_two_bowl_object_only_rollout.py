#!/usr/bin/env python3
"""Closed-loop bowl-region-only masks and per-step count-matched controls."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
CHECKPOINT = Path("/home/leju-suzhou/zjt_ws/checkpoints/libero/openvla-7b-finetuned-libero-spatial")
ARM_TO_MODE = {
    "target_only": "object_target_only",
    "distractor_only": "object_distractor_only",
    "non_bowl_target_count": "non_bowl_target_count",
    "non_bowl_distractor_count": "non_bowl_distractor_count",
}


def parse_ids(value: str) -> list[int]:
    ids: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            ids.extend(range(lo, hi + 1))
        else:
            ids.append(int(part))
    return sorted(set(ids))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True, help="physical inference GPU")
    ap.add_argument("--render-gpu", type=int, default=7)
    ap.add_argument("--arm", choices=tuple(ARM_TO_MODE), required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=220)
    args = ap.parse_args()

    visible = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    if args.gpu not in visible or args.render_gpu not in visible:
        raise RuntimeError(f"required GPUs inference={args.gpu}, render={args.render_gpu}; visible={visible}")
    inference_index = visible.index(args.gpu)

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    # Satisfy robosuite's import-time visibility assertion; then select EGL GPU
    # by the PCI-verified ordinal so rendering stays on physical GPU 7.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)
    from research.semantic_token_cd.resolve_mujoco_egl_device import resolve
    egl = resolve(args.render_gpu)
    from libero.libero.envs import OffScreenRenderEnv, SegmentationRenderEnv  # noqa: F401
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl["egl_ordinal"])

    import research.semantic_token_cd.libero_matched_rollout as rollout
    # Canonical task target is bowl_1 in both tasks; bowl_2 is the distractor.
    rollout.GT_TARGET_NAME = "akita_black_bowl_1"
    mode = ARM_TO_MODE[args.arm]
    sys.argv = [
        str(ROOT / "research/semantic_token_cd/libero_matched_rollout.py"),
        "--artifact", str(args.artifact.resolve()),
        "--checkpoint", str(CHECKPOINT),
        "--task", args.task,
        "--episodes", args.episodes,
        "--gpu", str(inference_index),
        "--suite", "libero_spatial",
        "--unnorm-key", "libero_spatial",
        "--entity-mode", "source_target",
        "--query-mode", "instruction_only",
        "--attention-layers", "11",
        "--position-mode", mode,
        "--selector-transform", "identity",
        "--max-steps", str(args.max_steps),
        "--env-seed", "0",
        "--settle-steps", "10",
    ]
    print(json.dumps({
        "event": "object_only_worker_start", "arm": args.arm,
        "position_mode": mode, "target_object": "akita_black_bowl_1",
        "distractor_object": "akita_black_bowl_2", "gpu": args.gpu,
        "render_gpu": args.render_gpu, "egl": egl, "task": args.task,
        "episodes": args.episodes,
    }, sort_keys=True), flush=True)
    rollout.main()

    anchor = "akita_black_bowl_2" if "distractor" in args.arm else "akita_black_bowl_1"
    for episode in parse_ids(args.episodes):
        path = args.artifact.resolve() / args.task / f"episode_{episode:03d}.json"
        result = json.loads(path.read_text())
        result["protocol_id"] = "LIBERO_SPATIAL_L11_OBJECT_ONLY_COUNT_CONTROL_V1"
        result["diagnostic_arm"] = args.arm
        result["anchor_object_name"] = anchor
        result["mask_scope"] = "selected tokens are restricted to the anchor object" if args.arm in ("target_only", "distractor_only") else "selected tokens exclude both bowl objects"
        result["count_reference_object"] = anchor if args.arm.startswith("non_bowl_") else None
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
