"""Offline extraction of full/entity/generic Top-P attention budgets.

For each frozen state this script stores the L11 visual attention vector for:
  * the complete task instruction
  * one phrase per extracted entity, using the locked template ``the {entity}``
  * the task's generic instruction

Only attention extraction is added here.  No closed-loop rollout and no success
labels are used to choose any threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE, get_image_from_maniskill2_obs_dict, restore_snapshot, snapshot_sha,
)
from research.semantic_token_cd.prompt_attn_layer_rollout import make_environment
from research.semantic_token_cd.prompt_attn_shr_policy_budget_estimators import (
    PromptAttentionSHRInference, _instruction_inputs, extract_prompt_attention,
    prompt_query_layout, N_VISUAL,
)
from research.semantic_token_cd.prompt_attn_shr_rollout import LAMBDA, atomic_json, load_reference
from research.semantic_token_cd.rollout_policy import extract_entities
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.target_specific_protocol import generic_instructions
from sklearn.cluster import KMeans

TASKS = ("google_robot_open_drawer", "google_robot_close_drawer",
         "google_robot_pick_coke_can", "google_robot_move_near")
TRAJ = {
    "google_robot_open_drawer": "artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes/google_robot_open_drawer/prompt_single",
    "google_robot_pick_coke_can": "artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes/google_robot_pick_coke_can/prompt_single",
    "google_robot_move_near": "artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes/google_robot_move_near/prompt_single",
    "google_robot_close_drawer": "artifacts/prompt_attn_layer_selection_v1/closed_loop_remaining6/episodes/google_robot_close_drawer/prompt_single",
}
PROGRESS = (0.10, 0.35, 0.60, 0.85)
KMEANS_K, KMEANS_SEED = 8, 0


def build_policy(base, task: str):
    p = base
    p.__class__ = PromptAttentionSHRInference
    _init_common(p, LAMBDA)
    p.beta = 0.0
    p.selector_mode = "prompt_attention"
    p.task_index = TASK_INDEX[task]
    p.attention_layers = (11,)
    p.kmeans_K, p.kmeans_seed = KMEANS_K, KMEANS_SEED
    p.save_prompt_attention = False
    return p


def normalize_attention(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (N_VISUAL,) or not np.isfinite(values).all():
        raise RuntimeError(f"invalid attention vector: {values.shape}")
    mass = float(values.sum())
    if not np.isfinite(mass) or mass <= 0:
        raise FloatingPointError("invalid attention mass")
    return values / mass


def phrase_attention(policy, image: np.ndarray, phrase: str, clean_visual) -> np.ndarray:
    inputs = _instruction_inputs(policy, image, phrase)
    scores, _ = extract_prompt_attention(
        policy, inputs, phrase, clean_visual, layers=(11,)
    )
    return normalize_attention(scores)


def matched_budget(visual: np.ndarray, instruction: str, tokenizer, policy):
    hd = visual.astype(np.float64)
    labels = KMeans(n_clusters=KMEANS_K, random_state=KMEANS_SEED, n_init=10).fit(hd).labels_
    group_vectors = np.stack([
        hd[labels == group].mean(axis=0) if np.any(labels == group) else np.zeros(hd.shape[1])
        for group in range(KMEANS_K)
    ])
    group_vectors /= np.linalg.norm(group_vectors, axis=1, keepdims=True) + 1e-8
    groups = []
    for entity in extract_entities(instruction):
        ids = tokenizer(entity, add_special_tokens=False)["input_ids"]
        if not ids:
            ids = tokenizer(entity, add_special_tokens=True)["input_ids"]
        import torch
        emb_t = torch.tensor([ids], dtype=torch.long, device=policy.vla.device)
        emb = policy.vla.language_model.model.embed_tokens(emb_t)[0].mean(dim=0).detach().float().cpu().numpy()
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        group = int(np.argmax(group_vectors @ emb))
        if group not in groups:
            groups.append(group)
    return int(sum(int(np.sum(labels == g)) for g in groups))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--canonical", type=Path, required=True)
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    repo = Path(__file__).resolve().parents[2]
    root = args.artifact.resolve()
    canonical = args.canonical.resolve()
    task = args.task
    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-", 1)); seeds.extend(range(lo, hi + 1))
        elif part.strip():
            seeds.append(int(part))
    seeds = sorted(set(seeds))

    out_dir = root / "states" / task
    out_dir.mkdir(parents=True, exist_ok=True)
    env, environment_id = make_environment(task, args.gpu)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, task, {}, False)
    policy = build_policy(OpenVLAInference(**config), task)
    tok = policy.processor.tokenizer
    special = set(int(v) for v in tok.all_special_ids)
    traj_dir = repo / TRAJ[task]

    for si, seed in enumerate(seeds):
        if args.shard_count > 1 and si % args.shard_count != args.shard_index:
            continue
        actions = np.load(traj_dir / f"episode_{seed:03d}_arrays.npz")["executed_actions"]
        ref = load_reference(canonical, task, seed)
        with (canonical / "snapshots" / task / f"seed_{seed:03d}.pkl").open("rb") as fh:
            snap = pickle.load(fh)
        if snapshot_sha(snap) != ref["canonical_snapshot_sha256"]:
            raise RuntimeError("canonical snapshot mismatch")
        obs, state_sha, rgb_sha = restore_snapshot(env, seed, snap)
        if (state_sha, rgb_sha) != (ref["initial_state_sha256"], ref["initial_rgb_sha256"]):
            raise RuntimeError("restored snapshot mismatch")
        instruction = env.unwrapped.get_language_instruction()
        L = len(actions)
        targets = sorted({min(L - 1, int(round((L - 1) * f))) for f in PROGRESS})

        for t in range(L):
            if t not in targets:
                obs = env.step(np.asarray(actions[t]))[0]
                continue
            started = time.monotonic()
            image = np.asarray(get_image_from_maniskill2_obs_dict(env, obs), dtype=np.uint8)
            inputs = policy.process_inputs(image, task_description=instruction)
            with projector_intervention(policy.vla) as trace:
                out = policy.vla(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"], use_cache=False,
                    output_attentions=True, return_dict=True,
                )
            visual = trace.before
            visual_np = visual[0].detach().float().cpu().numpy()  # [256,D]
            _, qpos, _ = prompt_query_layout(inputs["input_ids"], special)
            a_full = out.attentions[11][0, :, qpos, 1:1 + N_VISUAL].detach().float().cpu().mean(dim=(0, 1)).numpy()
            a_full = normalize_attention(a_full)
            entities = extract_entities(instruction)
            phrases = [f"the {e}" for e in entities]
            a_phrases = np.stack([phrase_attention(policy, image, phrase, visual) for phrase in phrases], axis=0)
            a_entity = normalize_attention(np.max(a_phrases, axis=0))
            generic = generic_instructions(task)[0]
            a_generic = phrase_attention(policy, image, generic, visual)
            m_matched = matched_budget(visual_np, instruction, tok, policy)
            stem = out_dir / f"seed_{seed:03d}_step_{t:03d}"
            tmp_npz = stem.with_name(stem.name + f".{os.getpid()}.tmp.npz")
            np.savez_compressed(
                tmp_npz,
                a_full=a_full.astype(np.float32),
                a_entity=a_entity.astype(np.float32),
                a_generic=a_generic.astype(np.float32),
                a_phrases=a_phrases.astype(np.float32),
                m_matched=np.asarray([m_matched], dtype=np.int64),
            )
            os.replace(tmp_npz, stem.with_suffix(".npz"))
            atomic_json(stem.with_suffix(".json"), {
                "task": task, "seed": int(seed), "step": int(t), "progress": float(t / max(1, L - 1)),
                "instruction": instruction, "entities": entities, "entity_phrases": phrases,
                "generic_instruction": generic, "m_matched": int(m_matched),
                "environment_id": environment_id,
                "canonical_snapshot_sha256": ref["canonical_snapshot_sha256"],
                "initial_state_sha256": ref["initial_state_sha256"],
                "initial_rgb_sha256": ref["initial_rgb_sha256"],
                "attention_sha256": {
                    "full": hashlib.sha256(np.ascontiguousarray(a_full).tobytes()).hexdigest(),
                    "entity": hashlib.sha256(np.ascontiguousarray(a_entity).tobytes()).hexdigest(),
                    "generic": hashlib.sha256(np.ascontiguousarray(a_generic).tobytes()).hexdigest(),
                },
                "runtime_seconds": time.monotonic() - started,
            })
            print(json.dumps({"task": task, "seed": int(seed), "step": int(t),
                              "m_matched": int(m_matched), "phrases": phrases,
                              "sec": round(time.monotonic() - started, 1)}), flush=True)
            obs = env.step(np.asarray(actions[t]))[0]


if __name__ == "__main__":
    main()
