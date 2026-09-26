#!/usr/bin/env python3
"""Diagnostic closed-loop arm that anchors each matched mask to one bowl.

The normal L11 attention ranking, matched m_t, harmonic reconstruction,
lambda, checkpoint, prompt, and greedy decoding remain unchanged. The only
selection change is that every visible patch token belonging to the selected
anchor bowl is prioritized, then any remaining slots up to m_t are filled by
the canonical L11 ranking. This is an oracle localization intervention, not a
deployable method.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
CHECKPOINT = Path("/home/leju-suzhou/zjt_ws/checkpoints/libero/openvla-7b-finetuned-libero-spatial")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True, help="physical inference GPU")
    ap.add_argument("--render-gpu", type=int, default=7)
    ap.add_argument("--arm", choices=("target_anchor", "distractor_anchor"), required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=220)
    args = ap.parse_args()

    visible = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    if args.gpu not in visible:
        raise RuntimeError(f"inference GPU {args.gpu} not visible in {visible}")
    inference_index = visible.index(args.gpu)
    if args.render_gpu not in visible:
        raise RuntimeError(f"renderer GPU {args.render_gpu} not visible in {visible}")

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    # Satisfy robosuite's import-time visibility assertion, then use the
    # PCI-verified EGL ordinal for the actual context.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)
    from research.semantic_token_cd.resolve_mujoco_egl_device import resolve
    egl = resolve(args.render_gpu)
    from libero.libero.envs import OffScreenRenderEnv, SegmentationRenderEnv  # noqa: F401
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl["egl_ordinal"])

    import research.semantic_token_cd.libero_matched_rollout as rollout
    anchor_name = "akita_black_bowl_1" if args.arm == "target_anchor" else "akita_black_bowl_2"
    rollout.GT_TARGET_NAME = anchor_name
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
        "--position-mode", "gt_target",
        "--selector-transform", "identity",
        "--max-steps", str(args.max_steps),
        "--env-seed", "0",
        "--settle-steps", "10",
    ]
    print(json.dumps({"event": "anchor_worker_start", "arm": args.arm, "anchor_object": anchor_name,
                      "gpu": args.gpu, "render_gpu": args.render_gpu, "egl": egl,
                      "task": args.task, "episodes": args.episodes}, sort_keys=True), flush=True)
    rollout.main()

    episode_ids = []
    for part in args.episodes.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            episode_ids.extend(range(lo, hi + 1))
        else:
            episode_ids.append(int(part))
    for episode in sorted(set(episode_ids)):
        path = args.artifact.resolve() / args.task / f"episode_{episode:03d}.json"
        result = json.loads(path.read_text())
        result["protocol_id"] = "LIBERO_SPATIAL_L11_OBJECT_ANCHOR_DIAGNOSTIC_V1"
        result["diagnostic_arm"] = args.arm
        result["anchor_object_name"] = anchor_name
        for step in result["trace"]:
            step["diagnostic_arm"] = args.arm
            step["anchor_object_name"] = anchor_name
            step["anchor_priority_token_count"] = step.pop("target_priority_token_count", 0)
            step["selected_anchor_token_count"] = step.pop("selected_target_count", 0)
            if args.arm == "distractor_anchor":
                step["selected_distractor_count"] = step["selected_anchor_token_count"]
                step["selected_target_count"] = None
            else:
                step["selected_distractor_count"] = 0
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
