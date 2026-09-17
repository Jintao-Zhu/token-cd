"""Run the fixed 60-state generic-frame and same-state action audit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_rollout import PCD_SOURCE
from research.semantic_token_cd.target_specific_common import build_policy
from research.semantic_token_cd.target_specific_protocol import (
    ARTIFACT, OFFLINE_ARMS, SOURCE_STATES, TASKS, atomic_json, generic_instructions, write_lock,
)


def chosen_states(task: str) -> list[Path]:
    roots = sorted((SOURCE_STATES / task).glob("seed_*"))
    if len(roots) < 5:
        raise RuntimeError(f"too few source episodes for {task}: {len(roots)}")
    positions = np.linspace(0, len(roots) - 1, 5).round().astype(int)
    output = []
    for index in positions:
        paths = sorted(roots[int(index)].glob("step_*.npz"))
        if len(paths) != 3:
            raise RuntimeError(f"expected three states in {roots[int(index)]}, got {len(paths)}")
        output.extend(paths)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--gpu", type=int, required=True); args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu); os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"; write_lock()
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    base = OpenVLAInference(**get_policy_config("openvla", checkpoint, args.task, {}, False))
    policies = {arm: build_policy(base, args.task, arm) for arm in OFFLINE_ARMS}
    out = ARTIFACT / "offline" / args.task; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in chosen_states(args.task):
        source_meta = json.loads(path.with_suffix(".json").read_text())
        image = np.asarray(np.load(path, allow_pickle=False)["image"], dtype=np.uint8)
        instruction = source_meta["instruction"]; branch = {}; arrays = {}
        clean_reference = None; m_reference = None
        for arm, policy in policies.items():
            policy.reset(instruction, seed=int(source_meta["seed"])); policy._episode_seed = int(source_meta["seed"])
            policy._selector_step = int(source_meta["control_step"]); policy._episode_trace = []; policy._episode_logits = []
            policy.step(image, None, instruction, proprio=np.zeros(8, dtype=np.float64))
            meta, record = policy._episode_trace[-1], policy._episode_logits[-1]
            clean = np.asarray(meta["clean_action"], dtype=np.float64)
            selected = np.asarray(meta["selected_token_ids"], dtype=np.int64); m = int(meta["m_t"])
            if clean_reference is None: clean_reference = clean
            if m_reference is None: m_reference = m
            cfg = {"m": m, "selected": selected.tolist(), "clean_action": clean.tolist(),
                   "guided_action": meta["guided_action"], "guided_clean_action_l2": meta["guided_clean_action_l2"],
                   "feature_perturbation_norm": meta["feature_perturbation_norm"],
                   "residual_norm": meta["centered_logit_residual_norm"],
                   "visual_attention_mass_correct": meta.get("correct_attention_mass_on_visual"),
                   "visual_attention_mass_generic": meta.get("contrast_attention_mass_on_visual")}
            if arm != "correct":
                p = np.asarray(record["correct_attention_probability"], dtype=np.float64)
                q = np.asarray(record["contrast_attention_probability"], dtype=np.float64)
                score = np.asarray(record["selector_score"], dtype=np.float64)
                cfg.update(selected_positive_fraction=float(np.mean(score[selected] > 0)),
                           mean_selected_correct_rank=float(np.mean(np.argsort(np.argsort(-p))[selected] + 1)),
                           mean_selected_p=float(p[selected].mean()), mean_selected_q=float(q[selected].mean()))
                arrays[f"{arm}_p"] = p; arrays[f"{arm}_q"] = q; arrays[f"{arm}_score"] = score
            mask = np.zeros(256, dtype=np.uint8); mask[selected] = 1; arrays[f"{arm}_mask"] = mask
            branch[arm] = cfg
        correct = set(branch["correct"]["selected"])
        for arm in OFFLINE_ARMS[1:]:
            selected = set(branch[arm]["selected"])
            branch[arm]["jaccard_vs_correct"] = len(correct & selected) / len(correct | selected)
        a, b = set(branch["target_diff"]["selected"]), set(branch["target_diff_alt"]["selected"])
        alt_jaccard = len(a & b) / len(a | b)
        row = {"task": args.task, "seed": source_meta["seed"], "control_step": source_meta["control_step"],
               "instruction": instruction, "generic_primary": generic_instructions(args.task)[0],
               "generic_alternate": generic_instructions(args.task)[1], "branches": branch,
               "clean_equal": all(np.allclose(clean_reference, x["clean_action"], atol=1e-6) for x in branch.values()),
               "m_equal": all(x["m"] == m_reference for x in branch.values()),
               "generic_paraphrase_mask_jaccard": alt_jaccard}
        stem = out / f"seed_{int(source_meta['seed']):03d}_step_{int(source_meta['control_step']):04d}"
        np.savez_compressed(stem.with_suffix(".npz"), **arrays); atomic_json(stem.with_suffix(".json"), row)
        rows.append(row); print(json.dumps({"task": args.task, "seed": row["seed"], "step": row["control_step"],
                                            "m": m_reference, "alt_jaccard": round(alt_jaccard, 3)}), flush=True)
    atomic_json(out / "SUMMARY.json", {"task": args.task, "states": len(rows),
        "clean_equal": all(x["clean_equal"] for x in rows), "m_equal": all(x["m_equal"] for x in rows)})


if __name__ == "__main__": main()
