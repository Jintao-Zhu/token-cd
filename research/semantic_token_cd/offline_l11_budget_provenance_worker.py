#!/usr/bin/env python3
"""Recover the provenance of L11 Matched budgets on saved same-state observations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.distractor_rollout import PCD_SOURCE
from research.semantic_token_cd.prompt_attn_l11_count_rollout import build_policies
from research.semantic_token_cd.prompt_attn_shr_policy import (
    extract_prompt_attention,
    stable_top_m,
)
from research.semantic_token_cd.prompt_attn_shr_rollout import atomic_json
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX
from research.semantic_token_cd.st_shr_policy import harmonic_reconstruct


TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
WRONG_ENTITIES = {
    "google_robot_open_drawer": ("coke can",),
    "google_robot_close_drawer": ("redbull can",),
    "google_robot_pick_coke_can": ("top drawer",),
    "google_robot_move_near": ("top drawer", "apple"),
}
ARMS = (
    "true_matched",
    "wrong_entity",
    "random_cluster",
    "source_only",
    "target_only",
    "top_p80",
    "true_scale_150",
)
RANDOM_SALT = 0xB0D6E7


def source_states(task: str) -> list[Path]:
    root = Path("artifacts/l11_matched_top_p_same_state_diagnostic_v1/same_state_eval")
    result = sorted((root / task).glob("seed_*/*.json"))
    if len(result) != 15:
        raise RuntimeError(f"expected 15 verified same-state replays for {task}, got {len(result)}")
    return result


def group_vectors(features: np.ndarray, labels: np.ndarray, group_count: int) -> np.ndarray:
    vectors = np.stack([features[labels == group].mean(axis=0) for group in range(group_count)])
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)


def groups_for_entities(policy, normalized_groups: np.ndarray, entities: tuple[str, ...]) -> list[int]:
    groups: list[int] = []
    for entity in entities:
        embedding = policy._embed_phrase(entity)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        group = int(np.argmax(normalized_groups @ embedding))
        if group not in groups:
            groups.append(group)
    return groups


def count_for_groups(labels: np.ndarray, groups: list[int]) -> int:
    return int(sum(np.count_nonzero(labels == group) for group in groups))


def top_p_count(scores: np.ndarray, threshold: float = 0.80) -> tuple[int, int]:
    probability = np.asarray(scores, dtype=np.float64)
    probability /= probability.sum()
    order = np.lexsort((np.arange(256), -probability))
    raw = int(np.searchsorted(np.cumsum(probability[order]), threshold) + 1)
    return raw, int(np.clip(raw, 16, 64))


def random_groups(task: str, seed: int, step: int, distinct_count: int, group_count: int) -> list[int]:
    random_seed = int(np.random.SeedSequence([
        TASK_INDEX[task], seed, step, RANDOM_SALT,
    ]).generate_state(1, dtype=np.uint32)[0])
    return sorted(int(value) for value in np.random.default_rng(random_seed).choice(
        group_count, size=distinct_count, replace=False,
    ))


def arm_metrics(features: np.ndarray, attention: np.ndarray, count: int) -> tuple[dict, np.ndarray]:
    selected = stable_top_m(attention, count)
    negative = features.copy()
    negative[selected] = harmonic_reconstruct(
        features, np.asarray(selected, dtype=np.int64), beta=0.0,
    )
    perturbation = negative - features
    mask = np.zeros(256, dtype=np.uint8)
    mask[selected] = 1
    probability = np.asarray(attention, dtype=np.float64)
    probability /= probability.sum()
    return {
        "count": count,
        "selected_attention_mass": float(probability[selected].sum()),
        "feature_perturbation_norm": float(np.linalg.norm(perturbation)),
        "feature_perturbation_relative": float(
            np.linalg.norm(perturbation) / (np.linalg.norm(features) + 1e-12)
        ),
    }, mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = args.artifact.resolve()
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    base = OpenVLAInference(**config)
    policy = build_policies(base, args.task, ("l11_matched",))["l11_matched"]

    for source_json in source_states(args.task):
        source = json.loads(source_json.read_text())
        source_arrays = np.load(source_json.with_suffix(".npz"))
        image = np.asarray(source_arrays["image"])
        instruction = source["instruction"]
        seed = int(source["seed"])
        step = int(source.get("source_step", source.get("step")))
        phase = source.get("phase")
        state_id = source.get("state_id", f"{args.task}__seed{seed:03d}__step{step:03d}")
        out_dir = artifact / "states" / args.task / f"seed_{seed:03d}"
        out_json = out_dir / f"step_{step:03d}.json"
        out_npz = out_json.with_suffix(".npz")
        if out_json.exists() and out_npz.exists():
            print(json.dumps({"skip": state_id}), flush=True)
            continue

        policy.reset(instruction, seed=seed)
        inputs = policy.process_inputs(image, task_description=instruction)
        with projector_intervention(policy.vla) as trace:
            clean_scores = policy._forward_scores(inputs, policy.unnorm_key, do_sample=False)
        if trace.before is None or clean_scores.shape[0] != 7:
            raise RuntimeError(f"clean replay failed: {state_id}")
        features = trace.before[0].numpy().astype(np.float32)
        attention, attention_meta = extract_prompt_attention(
            policy, inputs, instruction, trace.before, layers=(11,),
        )
        labels, _, per_entity_score = policy._semantic_clusters(features)
        true_groups = []
        for group in policy._entity_groups(features, labels):
            if group not in true_groups:
                true_groups.append(int(group))
        normalized_groups = group_vectors(features, labels, policy.kmeans_K)
        wrong_entities = WRONG_ENTITIES[args.task]
        wrong_groups = groups_for_entities(policy, normalized_groups, wrong_entities)
        random_group_ids = random_groups(
            args.task, seed, step, len(true_groups), policy.kmeans_K,
        )
        true_count = count_for_groups(labels, true_groups)
        expected_count = int(source["metrics"]["l11_matched"]["actual_selected_count"])
        if true_count != expected_count:
            raise RuntimeError(
                f"matched count replay mismatch for {state_id}: {true_count} != {expected_count}"
            )
        expected_attention = np.asarray(
            source_arrays["l11_matched__prompt_attention"], dtype=np.float32,
        )
        attention_max_abs_diff = float(np.max(np.abs(attention - expected_attention)))
        if attention_max_abs_diff != 0.0:
            raise RuntimeError(f"L11 attention replay mismatch for {state_id}: {attention_max_abs_diff}")

        raw_top_p80, clipped_top_p80 = top_p_count(attention)
        source_group = [true_groups[0]]
        target_group = [true_groups[-1]]
        counts = {
            "true_matched": true_count,
            "wrong_entity": count_for_groups(labels, wrong_groups),
            "random_cluster": count_for_groups(labels, random_group_ids),
            "source_only": count_for_groups(labels, source_group),
            "target_only": count_for_groups(labels, target_group),
            "top_p80": clipped_top_p80,
            "true_scale_150": int(np.clip(round(1.5 * true_count), 1, 256)),
        }
        metrics: dict[str, dict] = {}
        masks: dict[str, np.ndarray] = {}
        for arm in ARMS:
            metrics[arm], masks[arm] = arm_metrics(features, attention, counts[arm])
            true_mask = masks.get("true_matched")
            if true_mask is not None:
                intersection = int(np.logical_and(true_mask, masks[arm]).sum())
                union = int(np.logical_or(true_mask, masks[arm]).sum())
                metrics[arm]["jaccard_vs_true"] = intersection / max(1, union)

        group_sizes = [int(np.count_nonzero(labels == group)) for group in range(policy.kmeans_K)]
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_npz,
            labels=labels.astype(np.int16),
            group_sizes=np.asarray(group_sizes, dtype=np.int16),
            l11_attention=attention.astype(np.float32),
            **{f"mask__{arm}": masks[arm] for arm in ARMS},
        )
        atomic_json(out_json, {
            "protocol_id": "L11_MATCHED_BUDGET_PROVENANCE_OFFLINE_V1",
            "created_date": "2026-09-16",
            "state_id": state_id,
            "task": args.task,
            "seed": seed,
            "step": step,
            "phase": phase,
            "instruction": instruction,
            "true_entities": list(policy._entities),
            "wrong_entities": list(wrong_entities),
            "true_group_ids": true_groups,
            "wrong_group_ids": wrong_groups,
            "random_group_ids": random_group_ids,
            "source_group_id": source_group[0],
            "target_group_id": target_group[0],
            "entities_share_group": len(true_groups) == 1 and len(policy._entities) > 1,
            "per_entity_score": [float(value) for value in per_entity_score],
            "group_sizes": group_sizes,
            "top_p80_raw_count": raw_top_p80,
            "metrics": metrics,
            "technical": {
                "matched_count_exact": true_count == expected_count,
                "l11_attention_max_abs_diff": attention_max_abs_diff,
                "attention_layers": attention_meta["attention_layers"],
                "clean_score_rows": int(clean_scores.shape[0]),
                "source_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            },
            "arrays_file": out_npz.name,
            "source_json": str(source_json.resolve()),
        })
        print(json.dumps({
            "state": state_id,
            "counts": counts,
            "true_groups": true_groups,
            "wrong_groups": wrong_groups,
        }), flush=True)


if __name__ == "__main__":
    main()
