#!/usr/bin/env python3
"""SAME_STATE_FORK_V1 worker for LIBERO-90.

Restores a vanilla-trajectory state, then compares branches at that exact state:
  clean           official vanilla action
  reconstruction  independently decoded with harmonic reconstruction
  guided          frozen matched method
Each branch runs ``d`` intervention steps, then hands control to vanilla.

Also runs the acceptance checks that gate the main experiment.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from research.ar_token_counterfactual.intervention import decode_action_ids, projector_intervention
from research.ar_token_counterfactual.libero_runtime import (
    build_prompt,
    load_policy,
    predict_action,
    prepare_agentview,
    prepare_env_action,
    set_determinism,
)
from research.semantic_token_cd.libero_matched_rollout import (
    generate_clean_action,
    harmonic_reconstruct,
    predict_matched,
    prompt_attention_and_features,
    stable_top_m,
)
from research.semantic_token_cd.libero_policy import (
    entity_select,
    extract_source_target_entities_libero90,
    embed_phrase,
    forward_logits,
)
from research.semantic_token_cd.libero90_formal_worker import (
    atomic_json,
    claim_case,
    code_commit,
    complete_case,
    fail_case,
    sha256_file,
)
from research.semantic_token_cd.same_state_fork_common import (
    ATTENTION_LAYERS,
    K,
    KMEANS_SEED,
    LAMBDA,
    action_bin_margin,
    action_deltas,
    array_sha256,
    identity_reconstruction_baseline,
    resolve_duration,
)

PROTOCOL = "SAME_STATE_FORK_V1"
BRANCHES = ("clean", "reconstruction", "guided")


def make_env(bddl: str):
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)


def replay_to_step(env, init_state, actions, step: int):
    """Restore the init state and replay stored vanilla actions up to ``step``."""
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(10):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    for i in range(step):
        obs, _, _, _ = env.step(np.asarray(actions[i], dtype=float).tolist())
    return obs


def decode_reconstruction(model, processor, image, instruction, inputs, selected, replacement, unnorm_key):
    """Independently decode the reconstruction-only branch (no contrastive term)."""
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    eos_id = int(model.generation_config.eos_token_id)
    from transformers.generation.logits_process import LogitsProcessorList, SuppressTokensLogitsProcessor

    mean = torch.from_numpy(replacement).float()
    with projector_intervention(model, selected, mean) as trace:
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=inputs["pixel_values"],
            max_new_tokens=7,
            do_sample=False,
            logits_processor=LogitsProcessorList([SuppressTokensLogitsProcessor([eos_id])]),
        )
    if trace.before is None or trace.after is None:
        raise RuntimeError("reconstruction branch did not expose projector features")
    token_ids = generated[:, -7:]
    action = decode_action_ids(model, token_ids.detach().cpu(), unnorm_key)
    changed = [int(i) for i in trace.changed_indices]
    return action, token_ids[0].detach().cpu().tolist(), changed


def branch_actions(model, processor, image, instruction, inputs, clean_ids, clean_logits, clean_h, unnorm_key):
    """Compute the three branch actions and the same-prefix token diagnostics."""
    attention, h, attention_meta = prompt_attention_and_features(
        model, processor, image, instruction, inputs,
        expected_h=clean_h, query_mode="instruction_only",
        attention_layers=ATTENTION_LAYERS, attention_heads=(), destination_weight=0.0,
    )
    entities = extract_source_target_entities_libero90(instruction)
    if not entities:
        raise RuntimeError(f"no entities extracted: {instruction}")
    entity_embs = [embed_phrase(model, processor.tokenizer, e) for e in entities]
    selected_cluster, kmeans_meta = entity_select(h, entity_embs, K=K, seed=KMEANS_SEED)
    m = len(selected_cluster)
    selected = stable_top_m(attention, m)

    replacement = h.copy()
    replacement[np.asarray(selected, dtype=np.int64)] = harmonic_reconstruct(h, np.asarray(selected, dtype=np.int64))

    # Same-prefix teacher-forced negative logits (frozen guided branch).
    neg_logits = forward_logits(model, processor, image, instruction, clean_ids,
                                selected=selected, mean=torch.from_numpy(replacement).float())
    final = clean_logits.clone()
    final[:-1] = (1.0 + LAMBDA) * clean_logits[:-1] - LAMBDA * neg_logits[:-1]
    eos_id = int(model.generation_config.eos_token_id)
    final[:, eos_id] = torch.finfo(final.dtype).min
    guided_ids = final.argmax(dim=-1)
    guided_action = decode_action_ids(model, guided_ids.unsqueeze(0).detach().cpu(), unnorm_key)

    recon_action, recon_ids, recon_changed = decode_reconstruction(
        model, processor, image, instruction, inputs, selected, replacement, unnorm_key
    )

    action_start = int(model.vocab_size) - 256
    diagnostics = {
        "m_matched": int(m),
        "selected_token_ids": [int(x) for x in selected],
        "kmeans_groups": kmeans_meta.get("selected_groups"),
        "reconstruction_changed_token_ids": recon_changed,
        "reconstruction_only_modified_selected": bool(sorted(recon_changed) == sorted(int(x) for x in selected)),
        "feature_perturbation_relative": float(np.linalg.norm(replacement - h) / (np.linalg.norm(h) + 1e-12)),
        "clean_prefix_margin": action_bin_margin(clean_logits, action_start),
        "reconstruction_prefix_margin": action_bin_margin(neg_logits, action_start),
        "guided_prefix_margin": action_bin_margin(final, action_start),
        "clean_token_ids": [int(x) for x in clean_ids[0].detach().cpu().tolist()],
        "reconstruction_token_ids": recon_ids,
        "guided_token_ids": [int(x) for x in guided_ids.detach().cpu().tolist()],
        "attention_sha256": attention_meta.get("attention_sha256"),
    }
    return {
        "clean": np.asarray(decode_action_ids(model, clean_ids.detach().cpu(), unnorm_key), dtype=float),
        "reconstruction": np.asarray(recon_action, dtype=float),
        "guided": np.asarray(guided_action, dtype=float),
    }, diagnostics


def branch_step_action(branch, model, processor, image, instruction, inputs, unnorm_key):
    """One intervention step for the given branch, using the frozen method."""
    if branch == "clean":
        return predict_action(model, processor, image, instruction, unnorm_key=unnorm_key)
    if branch == "guided":
        action, _meta = predict_matched(
            model, processor, image, instruction, inputs=inputs,
            entity_mode="source_target_libero90", query_mode="instruction_only",
            attention_layers=ATTENTION_LAYERS, attention_heads=(), destination_weight=0.0,
            lambda_scale=1.0, unnorm_key=unnorm_key, position_mode="attention",
        )
        return np.asarray(action, dtype=float)
    if branch == "reconstruction":
        clean_ids, clean_logits, clean_h = generate_clean_action(model, processor, image, instruction, inputs)
        _attn, h, _meta = prompt_attention_and_features(
            model, processor, image, instruction, inputs,
            expected_h=clean_h, query_mode="instruction_only",
            attention_layers=ATTENTION_LAYERS, attention_heads=(), destination_weight=0.0,
        )
        entities = extract_source_target_entities_libero90(instruction)
        if not entities:
            raise RuntimeError(f"no entities extracted: {instruction}")
        entity_embs = [embed_phrase(model, processor.tokenizer, e) for e in entities]
        selected_cluster, _km = entity_select(h, entity_embs, K=K, seed=KMEANS_SEED)
        selected = stable_top_m(_attn, len(selected_cluster))
        replacement = h.copy()
        replacement[np.asarray(selected, dtype=np.int64)] = harmonic_reconstruct(
            h, np.asarray(selected, dtype=np.int64)
        )
        action, _ids, _changed = decode_reconstruction(
            model, processor, image, instruction, inputs, selected, replacement, unnorm_key
        )
        return np.asarray(action, dtype=float)
    raise ValueError(f"unknown branch {branch}")


def run_branch(env, branch, duration, init_state, actions, fork_step, model, processor, instruction,
               unnorm_key, max_steps, fork_image=None, fork_action=None, episode_len=None):
    """Execute ``duration`` intervention steps with the branch policy, then vanilla.

    The intervention is applied for the first ``duration`` decision steps.  Each
    intervention step re-derives the branch action from the *current* observation
    (that is what the deployed method does), it does not replay a frozen action.
    """
    obs = replay_to_step(env, init_state, actions, fork_step)
    # The fork state lies on a vanilla trajectory of length ``episode_len``.
    # The continuation must be bounded by that trajectory budget, not by the
    # runner's global step cap: otherwise the branch keeps stepping long after
    # the reference episode would have finished, which both wastes time and
    # makes the "remaining" duration meaningless.
    horizon = int(episode_len) if episode_len else int(max_steps)
    total_budget = max(1, horizon - fork_step)
    planned = resolve_duration(duration, total_budget)
    executed = 0
    done = False
    interventions = 0
    trace = []
    for i in range(total_budget):
        _, image = prepare_agentview(obs)
        if i == 0 and fork_action is not None:
            # step 0 uses the action computed from the shared fork image, so the
            # branch contrast is not contaminated by renderer noise.
            raw = fork_action
            source = branch
            interventions += 1 if branch != "clean" else 0
        elif i < planned and branch != "clean":
            inputs = processor(build_prompt(instruction), image).to(model.device, dtype=torch.bfloat16)
            raw = branch_step_action(branch, model, processor, image, instruction, inputs, unnorm_key)
            source = branch
            interventions += 1
        else:
            raw = predict_action(model, processor, image, instruction, unnorm_key=unnorm_key)
            source = "vanilla"
        env_action = prepare_env_action(raw)
        if env_action.shape != (7,) or not np.isfinite(env_action).all():
            raise FloatingPointError(f"invalid action {env_action}")
        if i < 3:
            trace.append({"i": i, "source": source, "action": np.asarray(raw, dtype=float).tolist()})
        obs, _reward, done, _info = env.step(env_action.tolist())
        executed += 1
        if done:
            break
    return {
        "success": bool(env.check_success()),
        "steps": executed,
        "intervention_steps_planned": int(planned),
        "intervention_steps_executed": int(interventions),
        "trace_head": trace,
    }


def run_case(case, suite, model, processor, args, config_hash, commit):
    from libero.libero import get_libero_path

    task_id = int(case["task_id"])
    task = suite.get_task(task_id)
    init_states = suite.get_task_init_states(task_id)
    init_state = init_states[int(case["init_state_id"])]
    bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)

    episode = json.loads(Path(case["episode_file"]).read_text())
    actions = np.asarray(episode["trajectory"], dtype=float)
    fork_step = int(case["fork_step"])

    env = make_env(bddl)
    try:
        obs = replay_to_step(env, init_state, actions, fork_step)
        state_sha = array_sha256(np.asarray(env.get_sim_state()))
        _, image = prepare_agentview(obs)
        image_sha = array_sha256(np.asarray(image))
        inputs = processor(build_prompt(task.language), image).to(model.device, dtype=torch.bfloat16)
        clean_ids, clean_logits, clean_h = generate_clean_action(model, processor, image, task.language, inputs)
        actions_by_branch, diagnostics = branch_actions(
            model, processor, image, task.language, inputs, clean_ids, clean_logits, clean_h, args.unnorm_key
        )
        diagnostics.update(identity_reconstruction_baseline(
            model, processor, image, task.language, inputs, clean_ids, clean_logits, args.unnorm_key
        ))

        deltas = {
            "recon_vs_clean": action_deltas(actions_by_branch["clean"], actions_by_branch["reconstruction"]),
            "guided_vs_clean": action_deltas(actions_by_branch["clean"], actions_by_branch["guided"]),
        }

        results = {}
        # One replay, one set of branch actions/diagnostics, then every requested
        # duration for every branch.  Replaying per-run was the dominant cost.
        durations = [int(x) for x in (case.get("durations") or [case["duration"]])]
        branches = case.get("branches") or [case.get("branch", "guided")]
        for branch in branches:
            for duration in durations:
                key = f"{branch}|d={duration}"
                results[key] = run_branch(
                    env, branch, duration, init_state, actions, fork_step,
                    model, processor, task.language, args.unnorm_key, args.max_steps,
                    fork_action=actions_by_branch[branch], episode_len=case["episode_len"],
                )
        if args.repeat_control:
            results["clean|repeat"] = run_branch(
                env, "clean", 1, init_state, actions, fork_step,
                model, processor, task.language, args.unnorm_key, args.max_steps,
                fork_action=actions_by_branch["clean"], episode_len=case["episode_len"],
            )
    finally:
        env.close()

    return {
        "protocol_id": PROTOCOL,
        "case_id": case["case_id"],
        "task_id": task_id,
        "task_name": task.name,
        "instruction": task.language,
        "init_state_id": int(case["init_state_id"]),
        "stage": case["stage"],
        "fork_step": fork_step,
        "episode_len": int(case["episode_len"]),
        "state_sha256": state_sha,
        "image_sha256": image_sha,
        "branch_actions": {k: v.tolist() for k, v in actions_by_branch.items()},
        "action_deltas": deltas,
        "diagnostics": diagnostics,
        "branches": list(branches),
        "durations": list(durations),
        "results": results,
        "checkpoint_revision": args.checkpoint_revision,
        "code_commit": commit,
        "config_sha256": config_hash,
        "worker_id": args.worker_id,
        "physical_gpu": int(args.physical_gpu),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--checkpoint-revision", required=True)
    ap.add_argument("--unnorm-key", default="libero_90_no_noops")
    ap.add_argument("--env-seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--physical-gpu", type=int, required=True)
    ap.add_argument("--render-gpu", type=int, default=None)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--durations", default="1,5,15,30,50,-1")
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--repeat-control", action="store_true")
    args = ap.parse_args()
    args.durations = tuple(int(x) for x in args.durations.split(",") if x)

    args.artifact = args.artifact.resolve()
    args.manifest = args.manifest.resolve()
    args.config = args.config.resolve()
    render = int(args.render_gpu or args.physical_gpu)

    visible = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x]
    if args.physical_gpu not in visible or render not in visible:
        raise RuntimeError(f"GPU mapping invalid visible={visible} model={args.physical_gpu} render={render}")
    if os.environ.get("MUJOCO_EGL_DEVICE_ID") != str(render):
        raise RuntimeError("MUJOCO_EGL_DEVICE_ID mismatch")
    model_index = visible.index(args.physical_gpu)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    from libero.libero import benchmark

    config_hash = sha256_file(args.config)
    commit = code_commit()
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    model, processor = load_policy(
        args.checkpoint,
        Path("/home/leju-suzhou/zjt_ws/token-cd/third_party/openvla/prismatic/extern/hf"),
        device=f"cuda:{model_index}",
        dataset_statistics_path=args.checkpoint / "dataset_statistics.json",
        unnorm_key=args.unnorm_key,
    )

    log = args.artifact / "logs" / f"worker_{args.worker_id}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    while args.max_cases <= 0 or processed < args.max_cases:
        claimed = claim_case(args.artifact, args.worker_id)
        if claimed is None:
            break
        case, running = claimed
        started = time.monotonic()
        try:
            set_determinism(20260923 + int(case["task_id"]) * 10000 + int(case["init_state_id"]))
            result = run_case(case, suite, model, processor, args, config_hash, commit)
            result["runtime_seconds"] = time.monotonic() - started
            atomic_json(args.artifact / "forks" / f"{case['case_id']}.json", result)
            complete_case(running, args.artifact)
            with log.open("a") as handle:
                handle.write(json.dumps({
                    "case_id": case["case_id"], "status": "complete",
                    "seconds": result["runtime_seconds"], "task_id": case["task_id"],
                    "stage": case["stage"], "fork_step": case["fork_step"],
                    "success": {k: v["success"] for k, v in result["results"].items()},
                }, sort_keys=True) + "\n")
        except Exception as exc:
            fail_case(running, args.artifact, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            with log.open("a") as handle:
                handle.write(json.dumps({
                    "case_id": case.get("case_id"), "status": "error",
                    "seconds": time.monotonic() - started, "error": f"{type(exc).__name__}: {exc}",
                }, sort_keys=True) + "\n")
        processed += 1


if __name__ == "__main__":
    main()
