"""XSWAP-V1 entry audit: 10 scenes/task single-step cross-arm check (no rollout).

For one restored state we compare the five arms on the identical image:
  - tokenization / query layout of actual vs paraphrase vs swapped instructions
  - coverage m identical across arms
  - clean action & clean feature identical regardless of selector instruction
  - selected masks recorded and pairwise Jaccard reported
Also measures per-step wall time.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_rollout import PCD_SOURCE, restore_snapshot
from research.semantic_token_cd.prompt_attn_shr_policy import N_VISUAL, PromptAttentionSHRInference
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.xswap_protocol import (
    ARMS, ARTIFACT, CANONICAL, LAMBDA, TASKS, instruction_set,
    resolve_present_phrases, swap_for_scene,
)
from research.semantic_token_cd.xswap_rollout import build_policy, make_environment


def jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--seeds", default="", help="comma list; empty -> manifest scenes")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--max-frames", type=int, default=1)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from parallel_inference import get_image_from_maniskill2_obs_dict
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    manifest = json.loads((ARTIFACT / "scene_manifest.json").read_text())["scenes"][args.task]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()] or manifest
    env, _env_id = make_environment(args.task, args.gpu)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    base = OpenVLAInference(**get_policy_config("openvla", checkpoint, args.task, {}, False))
    policies = {a: build_policy(base, args.task, a) for a in ARMS}

    report = {"task": args.task, "arms": [], "checks": {"passed": True, "problems": []}}
    for seed in seeds[:10]:
        with (CANONICAL / "snapshots" / args.task / f"seed_{seed:03d}.pkl").open("rb") as fh:
            snapshot = pickle.load(fh)
        obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
        instruction = env.unwrapped.get_language_instruction()
        present = resolve_present_phrases(env)
        plan = instruction_set(instruction)
        if plan["kind"] in ("pick_object", "move_near"):
            plan = swap_for_scene(instruction, present)
        image = get_image_from_maniskill2_obs_dict(env, obs)
        row = {"seed": seed, "instruction": instruction, "present": present,
               "plan": {k: plan[k] for k in plan if k in ("kind", "paraphrase", "swapped",
                                                          "swapped_kind", "swapped_target_phrase",
                                                          "source", "target_ref")},
               "selector": {}}
        clean_actions = {}
        for arm in ARMS:
            pol = policies[arm]
            if arm == "correct":
                pol.selector_instruction = None
            elif arm == "paraphrase":
                pol.selector_instruction = plan["paraphrase"]
            elif arm == "swapped":
                pol.selector_instruction = plan["swapped"]
            pol._episode_trace = []
            pol._episode_logits = []
            pol._episode_seed = seed
            pol._selector_step = 0
            pol.reset(instruction, seed=seed)
            t0 = time.monotonic()
            for _ in range(max(1, args.max_frames)):
                raw, actions, meta = pol.step(image, None, instruction, proprio=obs["agent"]["eef_pos"])
            dt = (time.monotonic() - t0) / max(1, args.max_frames)
            meta = pol._episode_trace[-1]
            if arm == "vanilla":
                from research.semantic_token_cd.rollout_pilot import flatten_action
                clean_actions[arm] = [float(x) for x in flatten_action(actions[0] if isinstance(actions, list) else actions)]
            else:
                selected = sorted(meta["selected_token_ids"])
                clean_actions[arm] = meta["clean_action"]
            row["selector"][arm] = {
                "m": int(meta["num_tokens"]) if arm != "vanilla" else None,
                "selected": selected if arm != "vanilla" else None,
                "attention_sha": meta.get("attention_sha256"),
                "query_tokens": meta.get("prompt_query_tokens"),
                "selector_instruction": meta.get("selector_instruction"),
                "selector_matches_task": meta.get("selector_matches_task"),
                "clean_action": clean_actions[arm],
                "per_step_seconds": round(dt, 3),
            }
        # checks
        inter = ("correct", "paraphrase", "swapped", "random")
        ms = [row["selector"][a]["m"] for a in inter]
        if len(set(ms)) != 1:
            report["checks"]["passed"] = False
            report["checks"]["problems"].append(f"seed {seed}: m differs {ms}")
        base_action = clean_actions["correct"]
        for arm in inter:
            if not np.allclose(clean_actions[arm], base_action, atol=1e-6):
                report["checks"]["passed"] = False
                report["checks"]["problems"].append(f"seed {seed}: clean action differs for {arm}")
        jacs = {}
        for arm in ("paraphrase", "swapped", "random"):
            jacs[arm] = round(jaccard(row["selector"]["correct"]["selected"],
                                      row["selector"][arm]["selected"]), 3)
        row["jaccard_vs_correct"] = jacs
        for arm in ("paraphrase", "swapped"):
            sm = row["selector"][arm]["selector_matches_task"]
            if sm is not False:
                report["checks"]["passed"] = False
                report["checks"]["problems"].append(f"seed {seed}: {arm} selector_matches_task={sm}")
        for arm in ("paraphrase", "swapped"):
            if row["selector"][arm]["query_tokens"] is None:
                report["checks"]["passed"] = False
                report["checks"]["problems"].append(f"seed {seed}: {arm} query layout missing")
        report["arms"].append(row)
    report["row_count"] = len(report["arms"])
    for arm in ARMS:
        seeds_arm = [r for r in report["arms"] if arm in r["selector"]]
        if not seeds_arm:
            continue
        per_step = np.mean([r["selector"][arm]["per_step_seconds"] for r in seeds_arm])
        report.setdefault("timing", {})[arm] = round(float(per_step), 3)
    out = ARTIFACT / "entry_check" / f"{args.task}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({
        "task": args.task, "rows": len(report["arms"]), "passed": report["checks"]["passed"],
        "problems": report["checks"]["problems"][:10], "timing": report.get("timing"),
    }, indent=1), flush=True)
    env.close()


if __name__ == "__main__":
    main()
