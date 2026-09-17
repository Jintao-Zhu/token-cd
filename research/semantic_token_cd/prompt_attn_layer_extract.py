"""Extract 0..31 full-prompt attention on the locked Prompt-v1 states.

This stage does not step the environment and does not evaluate candidate
layers.  It creates the immutable evidence used by the exploration/validation
split in the layer-selection experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.distractor_rollout import PCD_SOURCE
from research.semantic_token_cd.prompt_attn_shr_policy import (
    PromptAttentionSHRInference,
    extract_prompt_attention_per_layer,
    stable_top_m,
)
from research.semantic_token_cd.prompt_attn_shr_rollout import TASKS, build_policies


PROTOCOL = "PROMPT_ATTN_LAYER_SELECTION_V1"
SOURCE = Path("artifacts/prompt_attn_shr_failure_diagnosis_v1")

# Exactly 20 exploration episodes and 10 untouched validation episodes.  The
# split is episode-level: all early/middle/late states stay together.
VALIDATION = {
    "google_robot_open_drawer": {0, 1, 2},
    "google_robot_pick_coke_can": {3, 5, 95, 99},
    "google_robot_move_near": {0, 13, 89},
}


def rectangle(r0: int, r1: int, c0: int, c1: int) -> list[int]:
    return [r * 16 + c for r in range(r0, r1) for c in range(c0, c1)]


# A deliberately small set of approximate, human-inspected 16x16 reference
# boxes.  They diagnose whether attention moves with a changed target; they are
# never used by the selector and are not claimed as correct SHR masks.
TARGET_CONTROLS = {
    "google_robot_open_drawer__seed000__step011": {
        "new_instruction": "open middle drawer",
        "old_target": rectangle(5, 8, 2, 10),
        "new_target": rectangle(8, 10, 2, 10),
        "reference": "manual approximate drawer-front boxes",
    },
    "google_robot_open_drawer__seed011__step011": {
        "new_instruction": "open bottom drawer",
        "old_target": rectangle(7, 9, 7, 14),
        "new_target": rectangle(9, 11, 7, 14),
        "reference": "manual approximate drawer-front boxes",
    },
    "google_robot_open_drawer__seed032__step011": {
        "new_instruction": "open bottom drawer",
        "old_target": rectangle(7, 9, 0, 7),
        "new_target": rectangle(9, 11, 0, 7),
        "reference": "manual approximate drawer-front boxes",
    },
    "google_robot_pick_coke_can__seed003__step008": {
        "new_instruction": "pick pepsi can",
        "old_target": rectangle(6, 8, 9, 11),
        "new_target": rectangle(3, 5, 6, 8),
        "reference": "manual approximate object boxes",
    },
    "google_robot_pick_coke_can__seed030__step008": {
        "new_instruction": "pick pepsi can",
        "old_target": rectangle(6, 9, 6, 8),
        "new_target": rectangle(4, 7, 10, 12),
        "reference": "manual approximate object boxes",
    },
    "google_robot_pick_coke_can__seed059__step008": {
        "new_instruction": "pick pepsi can",
        "old_target": rectangle(7, 10, 3, 5),
        "new_target": rectangle(4, 7, 7, 9),
        "reference": "manual approximate object boxes",
    },
    "google_robot_move_near__seed000__step008": {
        "new_instruction": "move blue plastic bottle near 7up can",
        "old_target": rectangle(6, 9, 1, 4),
        "new_target": rectangle(6, 9, 9, 11),
        "reference": "manual approximate object boxes",
    },
    "google_robot_move_near__seed031__step008": {
        "new_instruction": "move 7up can near sponge",
        "old_target": rectangle(5, 7, 6, 8),
        "new_target": rectangle(8, 10, 9, 11),
        "reference": "manual approximate object boxes",
    },
    "google_robot_move_near__seed073__step008": {
        "new_instruction": "move sponge near apple",
        "old_target": rectangle(6, 9, 2, 4),
        "new_target": rectangle(5, 7, 6, 8),
        "reference": "manual approximate object boxes",
    },
}


def synonym(instruction: str) -> str:
    words = instruction.split()
    if instruction.startswith("open ") and instruction.endswith(" drawer"):
        return f"pull {' '.join(words[1:-1])} drawer open"
    if instruction == "pick coke can":
        return "grasp the coke can"
    if instruction.startswith("move ") and " near " in instruction:
        source, target = instruction[5:].split(" near ", 1)
        return f"place {source} close to {target}"
    raise ValueError(f"no locked synonym for: {instruction}")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_manifest(source: Path) -> dict:
    episodes = []
    for task in TASKS:
        for seed_dir in sorted((source / "states" / task).glob("seed_*")):
            seed = int(seed_dir.name.split("_")[-1])
            states = sorted(path.stem for path in seed_dir.glob("step_*.json"))
            split = "validation" if seed in VALIDATION[task] else "exploration"
            episodes.append({"task": task, "seed": seed, "split": split, "states": states})
    counts = {split: sum(row["split"] == split for row in episodes) for split in ("exploration", "validation")}
    if counts != {"exploration": 20, "validation": 10}:
        raise RuntimeError(f"unexpected split counts: {counts}")
    controls = []
    for state_id, control in TARGET_CONTROLS.items():
        task = state_id.split("__seed", 1)[0]
        seed = int(state_id.split("__seed", 1)[1].split("__", 1)[0])
        controls.append({
            "state_id": state_id,
            "task": task,
            "seed": seed,
            "split": "validation" if seed in VALIDATION[task] else "exploration",
            **control,
        })
    return {
        "protocol_id": PROTOCOL,
        "source": str(source.resolve()),
        "split_policy": "locked before layer metrics; episode-level; 20 exploration / 10 validation",
        "episodes": episodes,
        "target_controls": controls,
        "reference_box_boundary": "approximate diagnostic regions only; never used by selector",
    }


@torch.inference_mode()
def scores_for(policy, image: np.ndarray, instruction: str, clean_visual: torch.Tensor) -> tuple[np.ndarray, dict]:
    inputs = policy.process_inputs(image, task_description=instruction)
    return extract_prompt_attention_per_layer(policy, inputs, instruction, clean_visual)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
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
    artifact.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact / "SPLIT_AND_CONTROLS.json"
    manifest = build_manifest(source)
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("existing split/control manifest differs")
    atomic_json(manifest_path, manifest)

    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policy = build_policies(OpenVLAInference(**config), args.task)["prompt_attn_shr"]
    controls = {row["state_id"]: row for row in manifest["target_controls"]}
    episodes = [row for row in manifest["episodes"] if row["task"] == args.task]

    for episode in episodes:
        seed = episode["seed"]
        for stem in episode["states"]:
            source_json = source / "states" / args.task / f"seed_{seed:03d}" / f"{stem}.json"
            source_npz = source_json.with_suffix(".npz")
            state = json.loads(source_json.read_text())
            state_id = state["state_id"]
            out_dir = artifact / "states" / args.task / f"seed_{seed:03d}"
            out_json = out_dir / f"{stem}.json"
            out_npz = out_dir / f"{stem}.npz"
            if out_json.exists() and out_npz.exists():
                print(json.dumps({"skip": state_id}), flush=True)
                continue
            source_arrays = np.load(source_npz)
            image_array = source_arrays["image"]
            image = image_array
            instruction = state["instruction"]
            policy.reset(instruction, seed=seed)
            original_inputs = policy.process_inputs(image, task_description=instruction)
            with projector_intervention(policy.vla) as trace:
                clean_scores = policy._forward_scores(original_inputs, policy.unnorm_key, do_sample=False)
            if trace.before is None:
                raise RuntimeError(f"projector capture failed: {state_id}")
            clean_visual = trace.before
            original, meta = extract_prompt_attention_per_layer(
                policy, original_inputs, instruction, clean_visual
            )
            paraphrase = synonym(instruction)
            paraphrase_scores, paraphrase_meta = scores_for(policy, image, paraphrase, clean_visual)
            control = controls.get(state_id)
            target_scores = None
            target_meta = None
            if control is not None:
                target_scores, target_meta = scores_for(
                    policy, image, control["new_instruction"], clean_visual
                )
            m = int(state["m"])
            v1 = original[16:32].mean(axis=0)
            expected_v1 = source_arrays["attention"][2:4, 0].mean(axis=0)
            v1_max_abs_diff = float(np.max(np.abs(v1 - expected_v1)))
            if v1_max_abs_diff > 2e-6:
                raise RuntimeError(f"v1 attention reproduction mismatch {state_id}: {v1_max_abs_diff}")
            payload = {
                "original": original,
                "synonym": paraphrase_scores,
                "clean_positive": source_arrays["standard_shr__positive"],
                "standard_mask": source_arrays["standard_shr__mask"],
                "image": image_array,
            }
            if target_scores is not None:
                payload["target_switch"] = target_scores
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_npz, **payload)
            result = {
                "protocol_id": PROTOCOL,
                "state_id": state_id,
                "task": args.task,
                "seed": seed,
                "source_step": state["source_step"],
                "phase": state["phase"],
                "split": episode["split"],
                "instruction": instruction,
                "synonym_instruction": paraphrase,
                "target_control": control,
                "m": m,
                "v1_top_m": stable_top_m(v1, m),
                "v1_reproduction_max_abs_diff": v1_max_abs_diff,
                "source_rgb_sha256": hashlib.sha256(image_array.tobytes()).hexdigest(),
                "attention_meta": meta,
                "synonym_attention_meta": paraphrase_meta,
                "target_attention_meta": target_meta,
                "arrays_file": out_npz.name,
            }
            atomic_json(out_json, result)
            del clean_visual
            torch.cuda.empty_cache()
            print(json.dumps({"state": state_id, "split": episode["split"], "control": control is not None}), flush=True)


if __name__ == "__main__":
    main()
