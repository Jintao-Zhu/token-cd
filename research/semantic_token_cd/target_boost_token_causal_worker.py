"""Large same-state causal audit for Target Positive-Boost token swaps.

Every canonical Prompt-L11 trajectory contributes early/middle/late states.
At each state we recompute Correct and Positive-Boost-0.5 on the exact same
observation, then test each entered/exited token pair in two contexts:

* ``correct_with_swap_j``: apply only swap j to the Correct mask.
* ``boost_without_swap_j``: undo only swap j from the full Boost mask.

This preserves the matched token budget in every branch and separates an
individual swap's main effect from its interaction with the other swaps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE, get_image_from_maniskill2_obs_dict, restore_snapshot,
)
from research.semantic_token_cd.prompt_attn_l11_count_rollout import build_policies, make_environment
from research.semantic_token_cd.target_positive_boost_common import build_policy as build_boost_policy
from research.semantic_token_cd.target_specific_protocol import CANONICAL, TASKS
from research.semantic_token_cd.xswap_protocol import atomic_json


PROTOCOL = "TARGET_BOOST_TOKEN_CAUSAL_AUDIT_V1"
ARTIFACT = Path("/home/leju-suzhou/zjt_ws/token-cd/artifacts/target_boost_token_causal_audit_v1")
PROMPT_ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd/artifacts/prompt_attn_layer_selection_v1")
BOOST_ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd/artifacts/target_positive_boost_shr_v1")
PROMPT_SUBDIR = {
    "google_robot_open_drawer": "closed_loop",
    "google_robot_close_drawer": "closed_loop_remaining6",
    "google_robot_pick_coke_can": "closed_loop",
    "google_robot_move_near": "closed_loop",
}
FRACTIONS = (0.12, 0.50, 0.88)


def parse_seeds(spec: str) -> list[int]:
    values: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo, hi = map(int, item.split("-", 1)); values.extend(range(lo, hi + 1))
        else:
            values.append(int(item))
    result = sorted(set(values))
    if not result or min(result) < 0 or max(result) > 99:
        raise ValueError("seeds must be within 0..99")
    return result


def prompt_files(task: str, seed: int) -> tuple[Path, Path]:
    root = PROMPT_ROOT / PROMPT_SUBDIR[task] / "episodes" / task / "prompt_single"
    stem = f"episode_{seed:03d}"
    return root / f"{stem}_summary.json", root / f"{stem}_arrays.npz"


def boost_summary(task: str, seed: int) -> Path:
    return BOOST_ROOT / "closed_loop/episodes" / task / "positive_boost_0p5" / f"episode_{seed:03d}_summary.json"


def reset_policy(policy, instruction: str, seed: int, step: int) -> None:
    policy.reset(instruction, seed=seed)
    policy._episode_seed = seed
    policy._selector_step = step
    policy._episode_trace = []
    policy._episode_logits = []


def make_region_policy(base, task: str):
    policy = build_policies(base, task, ("l11_matched",))["l11_matched"]
    policy.selector_mode = "sim_region"
    policy.selection_count = None
    policy.save_prompt_attention = False
    return policy


def centered_residual(record: dict) -> np.ndarray:
    value = np.asarray(record["positive"][:6], dtype=np.float32) - np.asarray(
        record["negative"][:6], dtype=np.float32
    )
    return value - value.mean(axis=-1, keepdims=True)


def branch(policy, image: np.ndarray, instruction: str, proprio, seed: int, step: int,
           tokens: list[int] | None = None, label: str | None = None) -> tuple[dict, dict, np.ndarray]:
    reset_policy(policy, instruction, seed, step)
    if tokens is not None:
        policy.region_token_ids = list(tokens)
        policy.region_label = label
    _raw, _actions, meta = policy.step(image, None, instruction, proprio=proprio)
    record = policy._episode_logits[-1]
    residual = centered_residual(record)
    compact = {
        "selected_token_ids": [int(x) for x in meta["selected_token_ids"]],
        "positive_token_ids": [int(x) for x in meta["positive_token_ids"]],
        "negative_token_ids": [int(x) for x in meta["negative_token_ids"]],
        "final_token_ids": [int(x) for x in meta["final_token_ids"]],
        "clean_action": meta["clean_action"],
        "guided_action": meta["guided_action"],
        "guided_changed_dims": int(meta["guided_changed_dims"]),
        "feature_perturbation_norm": float(meta["feature_perturbation_norm"]),
        "centered_residual_norm": float(meta["centered_logit_residual_norm"]),
        "mask_component_count": int(meta["mask_component_count"]),
        "isolated_token_ratio": float(meta["isolated_token_ratio"]),
    }
    return compact, record, residual


def rank_map(score: np.ndarray) -> dict[int, int]:
    order = np.lexsort((np.arange(256), -np.asarray(score, dtype=np.float64)))
    return {int(token): int(rank + 1) for rank, token in enumerate(order)}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(np.dot(a, b) / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--worker-id", required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    base = OpenVLAInference(**get_policy_config("openvla", checkpoint, args.task, {}, False))
    correct = build_policies(base, args.task, ("l11_matched",))["l11_matched"]
    correct.save_prompt_attention = True
    boost = build_boost_policy(base, args.task, "positive_boost_0p5")
    region = make_region_policy(base, args.task)
    try:
        for seed in parse_seeds(args.seeds):
            output = ARTIFACT / "results" / args.task / f"seed_{seed:03d}.json"
            arrays_output = output.with_suffix(".npz")
            if output.exists() and arrays_output.exists():
                print(json.dumps({"skip": str(output)}), flush=True); continue
            prompt_summary_path, prompt_arrays_path = prompt_files(args.task, seed)
            if not (prompt_summary_path.exists() and prompt_arrays_path.exists() and boost_summary(args.task, seed).exists()):
                raise FileNotFoundError(f"missing completed source episode: {args.task}/{seed}")
            prompt_summary = json.loads(prompt_summary_path.read_text())
            boost_episode = json.loads(boost_summary(args.task, seed).read_text())
            historical = np.load(prompt_arrays_path, allow_pickle=False)
            actions = np.asarray(historical["executed_actions"], dtype=np.float64)
            targets = sorted(set(int(round((len(actions) - 1) * fraction)) for fraction in FRACTIONS))
            with (CANONICAL / "snapshots" / args.task / f"seed_{seed:03d}.pkl").open("rb") as handle:
                snapshot = pickle.load(handle)
            obs, initial_state_sha, initial_rgb_sha = restore_snapshot(env, seed, snapshot)
            if (initial_state_sha != prompt_summary["initial_state_sha256"]
                    or initial_rgb_sha != prompt_summary["initial_rgb_sha256"]):
                raise RuntimeError(f"initial identity mismatch: {args.task}/{seed}")
            instruction = env.unwrapped.get_language_instruction()
            state_rows, payload = [], {}
            for step in range(len(actions)):
                if step in targets:
                    image = np.asarray(get_image_from_maniskill2_obs_dict(env, obs), dtype=np.uint8)
                    proprio = obs["agent"]["eef_pos"]
                    c_meta, c_record, c_residual = branch(correct, image, instruction, proprio, seed, step)
                    b_meta, b_record, b_residual = branch(boost, image, instruction, proprio, seed, step)
                    c_tokens, b_tokens = set(c_meta["selected_token_ids"]), set(b_meta["selected_token_ids"])
                    if len(c_tokens) != len(b_tokens):
                        raise RuntimeError(f"matched budget violation: {args.task}/{seed}/{step}")
                    entered, exited = sorted(b_tokens - c_tokens), sorted(c_tokens - b_tokens)
                    if len(entered) != len(exited):
                        raise RuntimeError("swap cardinality mismatch")
                    p = np.asarray(b_record["correct_attention_probability"], dtype=np.float64)
                    q = np.asarray(b_record["contrast_attention_probability"], dtype=np.float64)
                    score = np.asarray(b_record["selector_score"], dtype=np.float64)
                    p_rank, q_rank, score_rank = rank_map(p), rank_map(q), rank_map(score)
                    # Pair the strongest entering token with the weakest retained Correct token.
                    entered.sort(key=lambda x: (-score[x], x))
                    exited.sort(key=lambda x: (p[x], x))
                    swaps, atomic_meta, atomic_residuals = [], {}, {}
                    for index, (token_in, token_out) in enumerate(zip(entered, exited)):
                        only = sorted((c_tokens - {token_out}) | {token_in})
                        undo = sorted((b_tokens - {token_in}) | {token_out})
                        only_name, undo_name = f"swap_{index:02d}_only", f"swap_{index:02d}_undone"
                        only_meta, _only_record, only_residual = branch(
                            region, image, instruction, proprio, seed, step, only, only_name
                        )
                        undo_meta, _undo_record, undo_residual = branch(
                            region, image, instruction, proprio, seed, step, undo, undo_name
                        )
                        atomic_meta[only_name], atomic_meta[undo_name] = only_meta, undo_meta
                        atomic_residuals[only_name], atomic_residuals[undo_name] = only_residual, undo_residual
                        swaps.append({
                            "index": index, "entered_token": token_in, "exited_token": token_out,
                            "entered_rc": [token_in // 16, token_in % 16],
                            "exited_rc": [token_out // 16, token_out % 16],
                            "entered_p": float(p[token_in]), "entered_q": float(q[token_in]),
                            "entered_difference": float(p[token_in] - q[token_in]),
                            "entered_p_rank": p_rank[token_in], "entered_q_rank": q_rank[token_in],
                            "entered_boost_rank": score_rank[token_in],
                            "exited_p": float(p[token_out]), "exited_q": float(q[token_out]),
                            "exited_difference": float(p[token_out] - q[token_out]),
                            "exited_p_rank": p_rank[token_out], "exited_q_rank": q_rank[token_out],
                            "exited_boost_rank": score_rank[token_out],
                            "only_residual_cosine_vs_correct": cosine(only_residual, c_residual),
                            "undo_residual_cosine_vs_boost": cosine(undo_residual, b_residual),
                            "only_action_l2_vs_correct": float(np.linalg.norm(
                                np.asarray(only_meta["guided_action"][:6]) - np.asarray(c_meta["guided_action"][:6])
                            )),
                            "undo_action_l2_vs_boost": float(np.linalg.norm(
                                np.asarray(undo_meta["guided_action"][:6]) - np.asarray(b_meta["guided_action"][:6])
                            )),
                        })
                    prefix = f"step_{step:03d}"
                    payload[f"{prefix}_image"] = image
                    payload[f"{prefix}_p"] = p
                    payload[f"{prefix}_q"] = q
                    payload[f"{prefix}_correct_mask"] = np.isin(np.arange(256), list(c_tokens)).astype(np.uint8)
                    payload[f"{prefix}_boost_mask"] = np.isin(np.arange(256), list(b_tokens)).astype(np.uint8)
                    payload[f"{prefix}_correct_residual"] = c_residual.astype(np.float16)
                    payload[f"{prefix}_boost_residual"] = b_residual.astype(np.float16)
                    for name, residual in atomic_residuals.items():
                        payload[f"{prefix}_{name}_residual"] = residual.astype(np.float16)
                    fraction_index = targets.index(step)
                    state_rows.append({
                        "step": step, "stage": ("early", "middle", "late")[fraction_index],
                        "fraction": float(step / max(1, len(actions) - 1)),
                        "image_sha256": hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
                        "m": len(c_tokens), "swap_count": len(swaps),
                        "historical_correct_mask_jaccard": float(len(c_tokens & set(np.flatnonzero(
                            historical["selected_mask"][step]
                        ))) / max(1, len(c_tokens | set(np.flatnonzero(historical["selected_mask"][step]))))),
                        "correct": c_meta, "boost": b_meta, "swaps": swaps,
                        "atomic_branches": atomic_meta,
                        "boost_residual_cosine_vs_correct": cosine(b_residual, c_residual),
                        "boost_action_l2_vs_correct": float(np.linalg.norm(
                            np.asarray(b_meta["guided_action"][:6]) - np.asarray(c_meta["guided_action"][:6])
                        )),
                        "boost_final_changed_dims_vs_correct": int(np.sum(
                            np.asarray(b_meta["final_token_ids"][:6]) != np.asarray(c_meta["final_token_ids"][:6])
                        )),
                    })
                if step < len(actions) - 1:
                    # Historical arrays already store the flattened 7-DoF
                    # action, not the policy's action dictionary.
                    obs, _reward, _terminated, truncated, _info = env.step(actions[step])
                    if truncated:
                        raise RuntimeError(f"historical replay truncated: {args.task}/{seed}/{step}")
            category = ("rescue" if not prompt_summary["result"]["success"] and boost_episode["success"] else
                        "harm" if prompt_summary["result"]["success"] and not boost_episode["success"] else
                        "both_success" if prompt_summary["result"]["success"] else "both_fail")
            arrays_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(arrays_output, **payload)
            atomic_json(output, {
                "protocol_id": PROTOCOL, "task": args.task, "seed": seed,
                "instruction": instruction, "generic_instruction": boost_episode["generic_instruction"],
                "outcome_category": category, "correct_success": bool(prompt_summary["result"]["success"]),
                "boost_success": bool(boost_episode["success"]), "states": state_rows,
                "arrays_file": str(arrays_output), "environment_id": environment_id,
                "initial_state_sha256": initial_state_sha, "initial_rgb_sha256": initial_rgb_sha,
                "worker_id": args.worker_id, "technical_pass": len(state_rows) == 3,
            })
            print(json.dumps({"complete": f"{args.task}/{seed}", "states": 3,
                              "category": category, "swaps": sum(x["swap_count"] for x in state_rows)}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
