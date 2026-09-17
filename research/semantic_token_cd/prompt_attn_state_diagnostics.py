"""Same-state mechanism diagnostics for Prompt-Attn-SHR v1.

Replays the completed standard-SHR trajectories from canonical snapshots and
evaluates Standard, Prompt-v1, and matched Random masks on identical images.
This script is intentionally diagnostic-only: it never steps the environment
with a newly computed action and therefore cannot alter closed-loop results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.distractor_policy import ACTION_VOCAB_SIZE, _action_logits
from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE,
    flatten_action,
    get_image_from_maniskill2_obs_dict,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.global_merge_policy import guided_forward_scores, projector_merge_intervention
from research.semantic_token_cd.prompt_attn_shr_policy import (
    EPS,
    LAYER_END,
    LAYER_START,
    N_VISUAL,
    RANDOM_SALT,
    prompt_query_layout,
    stable_top_m,
)
from research.semantic_token_cd.prompt_attn_shr_rollout import (
    KMEANS_K,
    KMEANS_SEED,
    LAMBDA,
    TASKS,
    build_policies,
    load_reference,
    make_environment,
)
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX
from research.semantic_token_cd.st_shr_policy import harmonic_reconstruct


ARMS = ("standard_shr", "prompt_attn_shr", "random_shr")
WINDOWS = ((0, 8), (8, 16), (16, 24), (24, 32))
QUERY_TYPES = ("full", "entity", "verb_relation")
RANDOM_REFERENCE_COUNT = 10
DIAGNOSTIC_SALT = 0xD1A69057


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def choose_evenly(values: list[int], count: int) -> list[int]:
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return values[:]
    positions = np.linspace(0, len(values) - 1, count)
    chosen = []
    for pos in positions:
        value = values[int(round(float(pos)))]
        if value not in chosen:
            chosen.append(value)
    for value in values:
        if len(chosen) == count:
            break
        if value not in chosen:
            chosen.append(value)
    return sorted(chosen)


def build_selection_manifest(closed_loop: Path) -> dict:
    """Select ten deterministic, outcome-stratified episodes per task."""
    manifest = {"policy": "outcome-stratified diagnostic sample; not prevalence-estimating", "tasks": {}}
    desired = {"standard_only": 4, "both_success": 2, "both_fail": 2, "prompt_only": 2}
    for task in TASKS:
        buckets = {key: [] for key in desired}
        for seed in range(100):
            standard = json.loads((closed_loop / "episodes" / task / "standard_shr" /
                                   f"episode_{seed:03d}_summary.json").read_text())["success"]
            prompt = json.loads((closed_loop / "episodes" / task / "prompt_attn_shr" /
                                 f"episode_{seed:03d}_summary.json").read_text())["success"]
            key = ("both_success" if standard and prompt else
                   "standard_only" if standard else
                   "prompt_only" if prompt else "both_fail")
            buckets[key].append(seed)
        chosen = []
        category = {}
        for key, target in desired.items():
            for seed in choose_evenly(buckets[key], min(target, len(buckets[key]))):
                chosen.append(seed); category[str(seed)] = key
        if len(chosen) < 10:
            remaining = [seed for seed in range(100) if seed not in chosen]
            for seed in choose_evenly(remaining, 10 - len(chosen)):
                standard = json.loads((closed_loop / "episodes" / task / "standard_shr" /
                                       f"episode_{seed:03d}_summary.json").read_text())["success"]
                prompt = json.loads((closed_loop / "episodes" / task / "prompt_attn_shr" /
                                     f"episode_{seed:03d}_summary.json").read_text())["success"]
                key = ("both_success" if standard and prompt else
                       "standard_only" if standard else
                       "prompt_only" if prompt else "both_fail")
                chosen.append(seed); category[str(seed)] = key
        manifest["tasks"][task] = {
            "seeds": sorted(chosen), "category": category,
            "available_counts": {key: len(values) for key, values in buckets.items()},
        }
    return manifest


def step_indices(length: int) -> list[int]:
    if length < 3:
        raise RuntimeError(f"trajectory too short for three states: {length}")
    raw = [int(round((length - 1) * fraction)) for fraction in (0.1, 0.5, 0.9)]
    result = []
    for value in raw:
        value = min(length - 1, max(0, value))
        while value in result and value + 1 < length:
            value += 1
        while value in result and value > 0:
            value -= 1
        result.append(value)
    return result


def find_subsequence(sequence: list[int], query: list[int]) -> list[int]:
    if not query:
        return []
    for start in range(len(sequence) - len(query) + 1):
        if sequence[start:start + len(query)] == query:
            return list(range(start, start + len(query)))
    return []


def query_sets(policy, input_ids: torch.Tensor, entities: list[str] | None = None) -> tuple[dict[str, list[int]], dict]:
    tokenizer = policy.processor.tokenizer
    special = set(int(value) for value in tokenizer.all_special_ids)
    text, full_mm, token_ids = prompt_query_layout(input_ids, special)
    ids = [int(value) for value in input_ids[0].detach().cpu().tolist()]
    entity_text = set()
    unmatched = []
    entities = list(policy._entities if entities is None else entities)
    for entity in entities:
        encoded = tokenizer(entity, add_special_tokens=False)["input_ids"]
        hits = find_subsequence(ids, [int(value) for value in encoded])
        if not hits:
            unmatched.append(entity)
        entity_text.update(hits)
    entity = [N_VISUAL + index for index in sorted(entity_text)]
    verb_relation = [position for index, position in zip(text, full_mm) if index not in entity_text]
    if not entity:
        entity = full_mm[:]
    if not verb_relation:
        verb_relation = full_mm[:]
    return {"full": full_mm, "entity": entity, "verb_relation": verb_relation}, {
        "text_indices": text,
        "multimodal_indices": full_mm,
        "token_ids": token_ids,
        "tokens": tokenizer.convert_ids_to_tokens(token_ids),
        "entity_multimodal_indices": entity,
        "verb_relation_multimodal_indices": verb_relation,
        "entities": entities,
        "unmatched_entities": unmatched,
    }


def instruction_counterfactual(task: str, instruction: str, entities: list[str]) -> tuple[str, list[str], str]:
    lower = instruction.lower()
    if task == "google_robot_open_drawer":
        return lower.replace("open", "close", 1), entities, "verb_swap_open_to_close"
    if task == "google_robot_pick_coke_can":
        return lower.replace("coke", "pepsi", 1), [e.replace("coke", "pepsi") for e in entities], "entity_swap_coke_to_pepsi"
    if task == "google_robot_move_near" and len(entities) >= 2:
        return f"move {entities[1]} near {entities[0]}", [entities[1], entities[0]], "source_target_swap"
    return instruction, entities, "identity_fallback"


@torch.inference_mode()
def attention_matrix(policy, inputs, queries: dict[str, list[int]], clean_visual: torch.Tensor):
    with projector_intervention(policy.vla) as trace:
        output = policy.vla(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"], use_cache=False,
            output_attentions=True, return_dict=True,
        )
    if trace.before is None or not torch.equal(trace.before, clean_visual):
        raise RuntimeError("attention forward changed projector output")
    if output.attentions is None or len(output.attentions) != 32:
        raise RuntimeError("expected 32 post-softmax language attention tensors")
    result = np.zeros((len(WINDOWS), len(QUERY_TYPES), N_VISUAL), dtype=np.float32)
    for wi, (lo, hi) in enumerate(WINDOWS):
        for qi, query_type in enumerate(QUERY_TYPES):
            chunks = [layer[0, :, queries[query_type], 1:1 + N_VISUAL].detach().float().cpu()
                      for layer in output.attentions[lo:hi]]
            result[wi, qi] = torch.stack(chunks).mean(dim=(0, 1, 2)).numpy()
    return result


def components(mask: np.ndarray) -> dict:
    selected = set(int(x) for x in np.flatnonzero(mask))
    sizes = []
    boundary_edges = 0
    seen = set()
    for token in selected:
        r, c = divmod(token, 16)
        neighbors = []
        if r: neighbors.append(token - 16)
        if r < 15: neighbors.append(token + 16)
        if c: neighbors.append(token - 1)
        if c < 15: neighbors.append(token + 1)
        boundary_edges += sum(neighbor not in selected for neighbor in neighbors)
        if token in seen:
            continue
        stack = [token]; seen.add(token); size = 0
        while stack:
            current = stack.pop(); size += 1
            rr, cc = divmod(current, 16)
            for neighbor in ((current - 16 if rr else -1), (current + 16 if rr < 15 else -1),
                             (current - 1 if cc else -1), (current + 1 if cc < 15 else -1)):
                if neighbor in selected and neighbor not in seen:
                    seen.add(neighbor); stack.append(neighbor)
        sizes.append(size)
    m = max(1, len(selected))
    corners = {0, 15, 240, 255}
    outer = {i for i in range(256) if i // 16 in (0, 15) or i % 16 in (0, 15)}
    return {
        "num_components": len(sizes),
        "isolated_token_ratio": sum(size == 1 for size in sizes) / m,
        "largest_component_ratio": max(sizes, default=0) / m,
        "boundary_edges_per_token": boundary_edges / m,
        "corner_ratio": len(selected & corners) / m,
        "outer_ring_ratio": len(selected & outer) / m,
    }


def evaluate_mask(policy, inputs, clean_scores, visual, h, selected: list[int]) -> tuple[dict, dict]:
    selected = sorted(set(int(value) for value in selected))
    negative_features = h.copy()
    negative_features[selected] = harmonic_reconstruct(h, np.asarray(selected), beta=0.0)
    with projector_merge_intervention(policy.vla, torch.from_numpy(negative_features).unsqueeze(0)):
        negative_scores = guided_forward_scores(policy.vla, inputs, clean_scores.argmax(-1), visual.shape[1])
    final_scores = clean_scores.clone()
    final_scores[:6] = 1.5 * clean_scores[:6] - 0.5 * negative_scores[:6]
    positive = _action_logits(policy, clean_scores).astype(np.float32)
    negative = _action_logits(policy, negative_scores).astype(np.float32)
    final = _action_logits(policy, final_scores).astype(np.float32)
    residual = positive[:6] - negative[:6]
    centered = residual - residual.mean(axis=1, keepdims=True)
    perturb = negative_features - h
    perturb_norm = np.linalg.norm(perturb, axis=1)
    base_norm = np.linalg.norm(h, axis=1)
    clean_local = positive.argmax(axis=1)
    guided_local = final.argmax(axis=1)
    clean_full = clean_scores.argmax(-1)
    guided_full = final_scores.argmax(-1)
    clean_action = policy._decode_actions(clean_full, policy.unnorm_key)
    guided_action = policy._decode_actions(guided_full, policy.unnorm_key)
    dims = []
    for q in range(6):
        a, b = int(clean_local[q]), int(guided_local[q])
        margin = float(positive[q, a] - positive[q, b])
        direction = float(residual[q, b] - residual[q, a])
        rank = int(1 + np.sum(positive[q] > positive[q, b]))
        dims.append({
            "dimension": q, "clean_winner": a, "guided_winner": b,
            "guided_winner_clean_rank": rank, "clean_margin": margin,
            "residual_advantage": direction, "lambda_residual_advantage": LAMBDA * direction,
            "winner_flipped": a != b,
            "inequality_predicts_flip": bool(a != b and LAMBDA * direction > margin),
            "clean_action_value": float(clean_action[q]),
            "guided_action_value": float(guided_action[q]),
            "action_delta": float(guided_action[q] - clean_action[q]),
        })
    mask = np.isin(np.arange(256), selected).astype(np.uint8)
    metric = {
        "m": len(selected), "selected_indices": selected, **components(mask),
        "feature_perturbation_total": float(np.linalg.norm(perturb)),
        "feature_perturbation_relative_global": float(np.linalg.norm(perturb) / (np.linalg.norm(h) + EPS)),
        "selected_mean_patch_perturbation": float(perturb_norm[selected].mean()),
        "selected_mean_relative_perturbation": float((perturb_norm[selected] / (base_norm[selected] + EPS)).mean()),
        "centered_residual_norm": float(np.linalg.norm(centered)),
        "centered_residual_norm_per_dim": np.linalg.norm(centered, axis=1).astype(float).tolist(),
        "winner_flip_count": int(np.sum(clean_local[:6] != guided_local[:6])),
        "clean_action": clean_action.astype(float).tolist(),
        "guided_action": guided_action.astype(float).tolist(),
        "dimensions": dims,
    }
    arrays = {
        "mask": mask, "perturbation_norm": perturb_norm.astype(np.float32),
        "positive": positive.astype(np.float32), "negative": negative.astype(np.float32),
        "final": final.astype(np.float32), "centered_residual": centered.astype(np.float32),
    }
    return metric, arrays


def random_mask(task: str, seed: int, step: int, draw: int, m: int) -> list[int]:
    rng_seed = int(np.random.SeedSequence([
        TASK_INDEX[task], seed, step, DIAGNOSTIC_SALT, draw,
    ]).generate_state(1, dtype=np.uint32)[0])
    return sorted(int(x) for x in np.random.default_rng(rng_seed).choice(256, m, replace=False))


def save_state(artifact: Path, task: str, seed: int, step: int, phase: str,
               category: str, instruction: str, image, policy, audit: bool,
               random_reference_count: int = RANDOM_REFERENCE_COUNT,
               expected_positive: np.ndarray | None = None,
               expected_standard_indices: list[int] | None = None) -> dict:
    state_id = f"{task}__seed{seed:03d}__step{step:03d}"
    state_dir = artifact / "states" / task / f"seed_{seed:03d}"
    state_dir.mkdir(parents=True, exist_ok=True)
    json_path = state_dir / f"step_{step:03d}.json"
    npz_path = state_dir / f"step_{step:03d}.npz"
    image_path = state_dir / f"step_{step:03d}.png"
    if json_path.exists() and npz_path.exists() and image_path.exists():
        return json.loads(json_path.read_text())

    inputs = policy.process_inputs(image, task_description=instruction)
    with projector_intervention(policy.vla) as clean_trace:
        clean_scores = policy._forward_scores(inputs, policy.unnorm_key, do_sample=False)
    visual = clean_trace.before
    if visual is None or clean_scores.shape[0] != 7:
        raise RuntimeError("clean branch capture failed")
    h = visual[0].numpy().astype(np.float32)
    labels, _, per_entity_score = policy._semantic_clusters(h)
    entity_groups = policy._entity_groups(h, labels)
    standard = sorted(set(int(i) for group in entity_groups for i in np.flatnonzero(labels == group)))
    m = len(standard)
    current_positive = _action_logits(policy, clean_scores)
    replay_audit = {}
    if expected_positive is not None:
        expected_positive = np.asarray(expected_positive, dtype=np.float16)
        replay_audit["clean_logits_max_abs_diff"] = float(np.max(np.abs(
            current_positive.astype(np.float32) - expected_positive.astype(np.float32)
        )))
        replay_audit["clean_greedy_matches_closed_loop"] = bool(np.array_equal(
            current_positive.argmax(axis=1), expected_positive.argmax(axis=1)
        ))
        if not replay_audit["clean_greedy_matches_closed_loop"]:
            raise RuntimeError(f"replay clean action mismatch: {state_id}")
    if expected_standard_indices is not None:
        replay_audit["standard_mask_matches_closed_loop"] = standard == sorted(
            int(value) for value in expected_standard_indices
        )
        if not replay_audit["standard_mask_matches_closed_loop"]:
            raise RuntimeError(f"replay standard mask mismatch: {state_id}")
    queries, query_meta = query_sets(policy, inputs["input_ids"])
    attention = attention_matrix(policy, inputs, queries, visual)
    prompt_scores = attention[2:4, 0].mean(axis=0)
    prompt = stable_top_m(prompt_scores, m)
    random_seed = int(np.random.SeedSequence([
        TASK_INDEX[task], seed, step, RANDOM_SALT,
    ]).generate_state(1, dtype=np.uint32)[0])
    random_selected = sorted(int(x) for x in np.random.default_rng(random_seed).choice(256, m, replace=False))
    selected = {"standard_shr": standard, "prompt_attn_shr": prompt, "random_shr": random_selected}
    metrics = {}; arrays = {}
    for arm in ARMS:
        metrics[arm], arrays[arm] = evaluate_mask(policy, inputs, clean_scores, visual, h, selected[arm])
    for arm in ("prompt_attn_shr", "random_shr"):
        inter = len(set(selected[arm]) & set(standard))
        metrics[arm]["standard_overlap"] = inter / m
        metrics[arm]["standard_jaccard"] = inter / len(set(selected[arm]) | set(standard))
        a = arrays[arm]["centered_residual"].ravel(); b = arrays["standard_shr"]["centered_residual"].ravel()
        metrics[arm]["standard_residual_cosine"] = (
            None if np.linalg.norm(a) < 1e-8 or np.linalg.norm(b) < 1e-8
            else float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        )
    random_refs = []
    for draw in range(random_reference_count):
        ref_metric, _ = evaluate_mask(policy, inputs, clean_scores, visual, h,
                                      random_mask(task, seed, step, draw, m))
        random_refs.append({key: ref_metric[key] for key in (
            "num_components", "isolated_token_ratio", "largest_component_ratio",
            "boundary_edges_per_token", "corner_ratio", "outer_ring_ratio",
            "feature_perturbation_total", "selected_mean_relative_perturbation",
            "centered_residual_norm", "winner_flip_count",
        )})
    audit_result = None
    counterfactual_attention = None
    counterfactual_meta = None
    if audit:
        second_attention = attention_matrix(policy, inputs, queries, visual)
        clean_again = policy._forward_scores(inputs, policy.unnorm_key, do_sample=False)
        audit_result = {
            "attention_independent_max_abs_diff": float(np.max(np.abs(second_attention - attention))),
            "clean_logits_max_abs_diff_after_attention": float(torch.max(torch.abs(clean_again - clean_scores)).item()),
            "clean_greedy_unchanged_after_attention": bool(torch.equal(clean_again.argmax(-1), clean_scores.argmax(-1))),
            "total_layers": 32, "used_layers": list(range(16, 32)), "attention_shape": list(attention.shape),
            "visual_key_start": 1, "visual_key_end_inclusive": 256,
        }
        variant, variant_entities, variant_kind = instruction_counterfactual(
            task, instruction, list(policy._entities)
        )
        variant_inputs = policy.process_inputs(image, task_description=variant)
        variant_queries, variant_query_meta = query_sets(
            policy, variant_inputs["input_ids"], entities=variant_entities
        )
        counterfactual_attention = attention_matrix(
            policy, variant_inputs, variant_queries, visual
        )
        counterfactual_meta = {
            "instruction": variant, "kind": variant_kind,
            "entities": variant_entities, "query": variant_query_meta,
        }
    rgb = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image, dtype=np.uint8)
    Image.fromarray(rgb).save(image_path)
    payload = {"image": rgb, "attention": attention, "labels": labels.astype(np.int16)}
    if counterfactual_attention is not None:
        payload["counterfactual_attention"] = counterfactual_attention
    for arm in ARMS:
        for key, value in arrays[arm].items():
            payload[f"{arm}__{key}"] = value
    np.savez_compressed(npz_path, **payload)
    result = {
        "protocol_id": "PROMPT_ATTN_SHR_SAME_STATE_DIAGNOSTICS_V1",
        "state_id": state_id, "task": task, "seed": seed,
        "source_policy": "standard_shr_diagnostic_rerun",
        "source_step": step, "phase": phase, "outcome_category": category,
        "instruction": instruction, "m": m, "query": query_meta,
        "attention_windows": [list(x) for x in WINDOWS], "query_types": list(QUERY_TYPES),
        "per_entity_score": per_entity_score, "selected_group_ids": entity_groups,
        "random_seed": random_seed, "metrics": metrics, "random_reference": random_refs,
        "audit": audit_result, "replay_audit": replay_audit,
        "counterfactual_instruction": counterfactual_meta,
        "image_file": str(image_path.relative_to(artifact)),
        "arrays_file": str(npz_path.relative_to(artifact)),
        "clean_positive_sha256": hashlib.sha256(current_positive.tobytes()).hexdigest(),
    }
    atomic_json(json_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--closed-loop-artifact", type=Path, required=True)
    parser.add_argument("--snapshot-artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seeds", default=None, help="optional comma-separated manifest subset")
    parser.add_argument("--random-reference-count", type=int, default=RANDOM_REFERENCE_COUNT)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact=args.artifact.resolve(); closed=args.closed_loop_artifact.resolve(); source=args.snapshot_artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    manifest_path=artifact/"selection_manifest.json"
    if not manifest_path.exists(): atomic_json(manifest_path, build_selection_manifest(closed))
    manifest=json.loads(manifest_path.read_text())
    env,_=make_environment(args.task,args.gpu)
    checkpoint=str(PCD_SOURCE/"pretrained/openvla-7b")
    config=get_policy_config("openvla",checkpoint,args.task,{},False)
    policies=build_policies(OpenVLAInference(**config),args.task)
    driver=policies["standard_shr"]
    policy=policies["prompt_attn_shr"]
    manifest_seeds=manifest["tasks"][args.task]["seeds"]
    requested=(set(int(x) for x in args.seeds.split(",")) if args.seeds else set(manifest_seeds))
    seeds=[seed for seed in manifest_seeds if seed in requested]
    if not seeds or any(seed not in manifest_seeds for seed in requested):
        raise ValueError("--seeds must be a non-empty subset of the locked diagnostic manifest")
    if args.random_reference_count < 0:
        raise ValueError("--random-reference-count must be non-negative")
    audit_seeds=set(manifest_seeds[:3])
    for seed in seeds:
        summary=json.loads((closed/"episodes"/args.task/"standard_shr"/f"episode_{seed:03d}_summary.json").read_text())
        with (source/"snapshots"/args.task/f"seed_{seed:03d}.pkl").open("rb") as handle: snapshot=pickle.load(handle)
        reference=load_reference(source,args.task,seed)
        if snapshot_sha(snapshot)!=reference["canonical_snapshot_sha256"]: raise RuntimeError("snapshot mismatch")
        obs,state_sha,rgb_sha=restore_snapshot(env,seed,snapshot)
        if (state_sha,rgb_sha)!=(reference["initial_state_sha256"],reference["initial_rgb_sha256"]): raise RuntimeError("restore mismatch")
        instruction=env.unwrapped.get_language_instruction()
        driver.reset(instruction,seed=seed); driver._episode_trace=[]; driver._episode_logits=[]
        cached_images=[]; cached_instructions=[]; predicted_terminated=False; truncated=False
        while not (predicted_terminated or truncated):
            current_instruction=env.unwrapped.get_language_instruction()
            image=get_image_from_maniskill2_obs_dict(env,obs)
            cached_images.append(image.copy() if hasattr(image,"copy") else np.asarray(image).copy())
            cached_instructions.append(current_instruction)
            _raw,actions,_meta=driver.step(image,None,current_instruction,proprio=obs["agent"]["eef_pos"])
            if not isinstance(actions,list): actions=[actions]
            for action in actions:
                executed=flatten_action(action)
                obs,_reward,_success,truncated,_info=env.step(executed)
                predicted_terminated=bool(action["terminate_episode"][0]>0)
                if predicted_terminated and not env.unwrapped.is_final_subtask():
                    predicted_terminated=False; env.advance_to_next_subtask()
        if len(cached_images)!=len(driver._episode_trace) or len(cached_images)!=len(driver._episode_logits):
            raise RuntimeError("diagnostic Standard trajectory trace length mismatch")
        original_trace=summary["selector_trace"]
        if driver._episode_trace[0]["positive_token_ids"]!=original_trace[0]["positive_token_ids"]:
            raise RuntimeError(f"diagnostic rerun initial clean action mismatch: {args.task} {seed}")
        if driver._episode_trace[0]["selected_token_ids"]!=original_trace[0]["selected_token_ids"]:
            raise RuntimeError(f"diagnostic rerun initial Standard mask mismatch: {args.task} {seed}")
        targets=step_indices(len(cached_images)); phases=dict(zip(targets,("early","middle","late")))
        policy.reset(instruction,seed=seed)
        category=manifest["tasks"][args.task]["category"][str(seed)]
        for step in targets:
            policy._selector_step=step
            result=save_state(
                artifact,args.task,seed,step,phases[step],category,cached_instructions[step],
                cached_images[step],policy,
                audit=(seed in audit_seeds and phases[step]=="early"),
                random_reference_count=args.random_reference_count,
                expected_positive=driver._episode_logits[step]["positive"],
                expected_standard_indices=driver._episode_trace[step]["selected_token_ids"],
            )
            print(json.dumps({"state":result["state_id"],"audit":result["audit"] is not None}),flush=True)
    print(json.dumps({"task":args.task,"complete":True}),flush=True)


if __name__ == "__main__":
    main()
