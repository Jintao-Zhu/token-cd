from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from research.ar_token_counterfactual.intervention import (
    action_logit_metrics,
    clean_action_token_ids,
    teacher_forced_forward,
    tensor_sha256,
)
from research.ar_token_counterfactual.libero_runtime import build_prompt, load_policy, prepare_agentview, set_determinism

from .common import array_sha256, make_env, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    plans = [row for i, row in enumerate(read_jsonl(artifact / "snapshot_plan.jsonl")) if i % args.shard_count == args.shard_index]
    source = workspace / "artifacts/ar_token_counterfactual_qualification_v1_20260808_231044"
    replacement = torch.load(source / "position_conditioned_visual_mean.pt", map_location="cpu", weights_only=True)["mean"]
    checkpoint = workspace / "checkpoints/openvla-7b-finetuned-libero-spatial/962318cec55ac10993ff0f5f43eda9a270b4c873"
    model, processor = load_policy(checkpoint, workspace / "third_party/openvla/prismatic/extern/hf")
    set_determinism(7)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    for ordinal, plan in enumerate(plans, 1):
        output = artifact / "candidates" / f"{plan['snapshot_id']}.json"
        if output.exists():
            continue
        snapshot = json.loads((artifact / "snapshots" / plan["snapshot_id"] / "record.json").read_text())
        state = np.load(artifact / snapshot["sim_state_path"], allow_pickle=False)
        task = suite.get_task(plan["task_id"])
        env = make_env(task, get_libero_path, OffScreenRenderEnv)
        try:
            env.reset()
            obs = env.set_init_state(state)
            restored = np.asarray(env.get_sim_state()).copy()
            _, image = prepare_agentview(obs)
        finally:
            env.close()
        if array_sha256(restored) != snapshot["sim_state_sha256"]:
            raise RuntimeError(f"Simulator state restoration mismatch: {plan['snapshot_id']}")

        inputs = processor(build_prompt(task.language), image).to(model.device, dtype=torch.bfloat16)
        protected = {key: tensor_sha256(inputs[key]) for key in ("input_ids", "attention_mask", "pixel_values")}
        clean_ids = clean_action_token_ids(model, inputs)
        clean = teacher_forced_forward(model, inputs, clean_ids)
        if clean.visual_token_count != 256 or tuple(replacement.shape) != (256, clean.trace.before.shape[-1]):
            raise RuntimeError("Locked visual-token or replacement shape mismatch")
        effects = []
        for token_index in range(256):
            masked = teacher_forced_forward(model, inputs, clean_ids, [token_index], replacement)
            metrics = action_logit_metrics(clean.logits, masked.logits)
            js = np.asarray(metrics["js_div"])[0]
            if masked.trace.changed_indices != (token_index,) or not np.isfinite(js).all():
                raise RuntimeError(f"Candidate integrity failure at {plan['snapshot_id']} token {token_index}")
            effects.append({"token_index": token_index, "mean_teacher_forced_js": float(js.mean()), "per_action_js": js.tolist()})
        if any(tensor_sha256(inputs[key]) != digest for key, digest in protected.items()):
            raise RuntimeError("Protected input mutation")
        ordered_max = sorted(effects, key=lambda row: (-row["mean_teacher_forced_js"], row["token_index"]))
        ordered_near = sorted(effects, key=lambda row: (row["mean_teacher_forced_js"], row["token_index"]))
        max_index, near_index = ordered_max[0]["token_index"], ordered_near[0]["token_index"]
        seed = 20260809 + int(hashlib.sha256(plan["snapshot_id"].encode()).hexdigest()[:8], 16)
        eligible = [index for index in range(256) if index not in (max_index, near_index)]
        random_index = int(np.random.default_rng(seed).choice(eligible))
        by_index = {row["token_index"]: row for row in effects}
        write_json(output, {
            **plan,
            "task_description": task.language,
            "restored_state_sha256": array_sha256(restored),
            "clean_action_token_ids": clean_ids[0].cpu().tolist(),
            "clean_projector_sha256": tensor_sha256(clean.trace.before),
            "selection_seed": seed,
            "selected": {
                "mask_max_effect": by_index[max_index],
                "mask_near_zero": by_index[near_index],
                "mask_random": by_index[random_index],
            },
            "effects": effects,
            "all_finite": True,
        })
        print(json.dumps({"shard": args.shard_index, "ordinal": ordinal, "snapshot_id": plan["snapshot_id"], "complete": True}), flush=True)


if __name__ == "__main__":
    main()
