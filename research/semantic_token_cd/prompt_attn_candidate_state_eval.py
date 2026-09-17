"""Same-state reconstruction/logit/action audit for frozen layer candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.distractor_policy import _action_logits
from research.semantic_token_cd.distractor_rollout import PCD_SOURCE
from research.semantic_token_cd.prompt_attn_shr_policy import stable_top_m
from research.semantic_token_cd.prompt_attn_shr_rollout import TASKS, build_policies
from research.semantic_token_cd.prompt_attn_state_diagnostics import evaluate_mask


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = args.artifact.resolve()
    source = args.source.resolve()
    lock = json.loads((artifact / "CANDIDATES_LOCK.json").read_text())
    configs = {
        "prompt_single": tuple(lock["prompt_single"]),
        "prompt_sparse": tuple(lock["prompt_sparse"]),
    }
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policy = build_policies(OpenVLAInference(**config), args.task)["prompt_attn_shr"]

    for layer_json in sorted((artifact / "states" / args.task).glob("*/*.json")):
        row = json.loads(layer_json.read_text())
        out_dir = artifact / "candidate_state_eval" / args.task / layer_json.parent.name
        out_json = out_dir / layer_json.name
        out_npz = out_json.with_suffix(".npz")
        if out_json.exists() and out_npz.exists():
            print(json.dumps({"skip": row["state_id"]}), flush=True)
            continue
        layer_arrays = np.load(layer_json.with_suffix(".npz"))
        image = layer_arrays["image"]
        instruction = row["instruction"]
        seed = int(row["seed"])
        policy.reset(instruction, seed=seed)
        inputs = policy.process_inputs(image, task_description=instruction)
        with projector_intervention(policy.vla) as trace:
            clean_scores = policy._forward_scores(inputs, policy.unnorm_key, do_sample=False)
        if trace.before is None:
            raise RuntimeError(f"projector capture failed: {row['state_id']}")
        visual = trace.before
        h = visual[0].numpy().astype(np.float32)
        positive = _action_logits(policy, clean_scores).astype(np.float32)
        positive_diff = float(np.max(np.abs(positive - layer_arrays["clean_positive"].astype(np.float32))))
        if not np.array_equal(positive.argmax(axis=1), layer_arrays["clean_positive"].argmax(axis=1)):
            raise RuntimeError(f"clean greedy mismatch: {row['state_id']}")
        metrics = {}
        payload = {"image": image, "positive": positive}
        for name, layers in configs.items():
            scores = layer_arrays["original"][list(layers)].mean(axis=0)
            selected = stable_top_m(scores, int(row["m"]))
            metric, arrays = evaluate_mask(policy, inputs, clean_scores, visual, h, selected)
            metrics[name] = metric
            payload[f"{name}__attention"] = scores.astype(np.float32)
            for key, value in arrays.items():
                payload[f"{name}__{key}"] = value
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_npz, **payload)
        atomic_json(out_json, {
            "protocol_id": lock["protocol_id"],
            "state_id": row["state_id"], "task": args.task, "seed": seed,
            "source_step": row["source_step"], "phase": row["phase"], "split": row["split"],
            "instruction": instruction, "m": row["m"], "configs": {k: list(v) for k, v in configs.items()},
            "metrics": metrics, "clean_positive_max_abs_diff": positive_diff,
            "clean_greedy_reproduced": True,
            "candidate_lock_sha256": hashlib.sha256((artifact / "CANDIDATES_LOCK.json").read_bytes()).hexdigest(),
            "arrays_file": out_npz.name,
        })
        del visual
        torch.cuda.empty_cache()
        print(json.dumps({"state": row["state_id"], "split": row["split"]}), flush=True)


if __name__ == "__main__":
    main()
