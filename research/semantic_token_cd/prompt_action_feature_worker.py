"""Recompute projector features for the locked 240 states and quantify mask geometry."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.prompt_action_complement_protocol import ARTIFACT, PCD_SOURCE, STATE_SOURCE, TASKS, atomic_json
from research.semantic_token_cd.prompt_action_complement_state import build_policy
from research.semantic_token_cd.st_shr_policy import harmonic_reconstruct


def cosine_cross(a, b):
    if len(a) == 0 or len(b) == 0: return None
    aa = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    bb = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return float((aa @ bb.T).mean())


def cosine_internal(a):
    if len(a) < 2: return None
    aa = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    matrix = aa @ aa.T
    return float(matrix[~np.eye(len(a), dtype=bool)].mean())


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--gpu", choices=(2, 3), type=int, required=True)
    parser.add_argument("--shard-index", type=int, default=0); parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu); os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    base = OpenVLAInference(**get_policy_config("openvla", str(PCD_SOURCE / "pretrained/openvla-7b"), args.task, {}, False))
    policy = build_policy(base, args.task)
    items = sorted((ARTIFACT / "stage_b/states" / args.task).glob("seed_*/*.json"))
    items = [item for index, item in enumerate(items) if index % args.num_shards == args.shard_index]
    for item in items:
        relative = item.relative_to(ARTIFACT / "stage_b/states" / args.task)
        output = ARTIFACT / "mechanism_analysis/features" / args.task / relative
        if output.exists(): continue
        meta = json.loads(item.read_text())
        source = STATE_SOURCE / "runs/emitted_states" / args.task / relative.with_suffix(".npz")
        image = np.asarray(np.load(source)["image"], dtype=np.uint8)
        inputs = policy.process_inputs(image, task_description=meta["instruction"])
        with projector_intervention(policy.vla) as trace:
            policy.vla(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                       pixel_values=inputs["pixel_values"], use_cache=False, return_dict=True)
        if trace.before is None: raise RuntimeError("projector features unavailable")
        features = trace.before[0].numpy().astype(np.float32)
        results = {}
        for arm, spec in meta["metrics"].items():
            selected = np.asarray(spec["selected"], dtype=int); core = np.asarray(spec["core"], dtype=int)
            supplement = np.asarray(spec["supplement"], dtype=int)
            reconstructed = harmonic_reconstruct(features, selected, beta=0.0)
            errors = np.linalg.norm(reconstructed - features[selected], axis=1)
            location = {token: index for index, token in enumerate(selected.tolist())}
            core_errors = [errors[location[int(token)]] for token in core]
            supplement_errors = [errors[location[int(token)]] for token in supplement]
            results[arm] = {
                "supplement_core_feature_cosine": cosine_cross(features[supplement], features[core]),
                "supplement_internal_feature_cosine": cosine_internal(features[supplement]),
                "core_internal_feature_cosine": cosine_internal(features[core]),
                "selected_reconstruction_error_mean": float(errors.mean()),
                "selected_reconstruction_error_norm": float(np.linalg.norm(errors)),
                "core_reconstruction_error_mean": float(np.mean(core_errors)),
                "supplement_reconstruction_error_mean": float(np.mean(supplement_errors)) if len(supplement_errors) else None,
            }
        atomic_json(output, {"task": args.task, "seed": meta["seed"], "control_step": meta["control_step"],
                             "arms": results, "source_state": str(source)})
        print(json.dumps({"task": args.task, "seed": meta["seed"], "step": meta["control_step"]}), flush=True)


if __name__ == "__main__": main()

