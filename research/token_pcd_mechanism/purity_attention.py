from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from research.ar_token_counterfactual.intervention import teacher_forced_forward
from research.token_pcd_mechanism.early_vs_late import load_inputs, replace_patch_tokens
from research.token_pcd_stage_a.core import (action_token_slice, cosine, direction_metrics, pcd_ids,
                                              read_jsonl, teacher_forced_logits)


def stable_top(scores, candidates, count):
    return sorted(sorted(candidates, key=lambda index: (-float(scores[index]), index))[:count])


def branch(model, inputs, clean_ids, selected, means):
    with replace_patch_tokens(model, selected, means):
        logits, _ = teacher_forced_logits(model, inputs, clean_ids)
    if not torch.isfinite(logits).all():
        raise RuntimeError("Early-intervention branch produced non-finite logits")
    return logits


def metrics(model, clean, pixel, value):
    pixel_delta = direction_metrics(model, clean, pixel)
    delta = direction_metrics(model, clean, value)
    pixel_norm = torch.linalg.vector_norm(pixel_delta.double())
    norm = torch.linalg.vector_norm(delta.double())
    return {"cosine": cosine(delta, pixel_delta), "effect_norm": float(norm),
            "pixel_norm_ratio": float(norm / pixel_norm) if pixel_norm else float("nan")}


def bootstrap_difference(rows, left, right, seed, draws=20000):
    paired = [(float(row[left]), float(row[right])) for row in rows
              if np.isfinite(row[left]) and np.isfinite(row[right])]
    if not paired:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed); n = len(paired)
    values = np.empty(draws)
    for index in range(draws):
        sample = rng.integers(0, n, n)
        values[index] = np.median([paired[i][0] for i in sample]) - np.median([paired[i][1] for i in sample])
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def finite_median(rows, name):
    values = [float(row[name]) for row in rows if np.isfinite(row[name])]
    return float(np.median(values)) if values else float("nan")


def summarize(results, artifact):
    flat = [{"task": row["task"], **{name: row["branches"][name]["cosine"] for name in row["branches"]},
             "norm_ratio": row["branches"]["pure_attention"]["pixel_norm_ratio"],
             "k": row["selected_token_count"]} for row in results]
    names = ("early25", "pure80", "pure_attention", "matched_random", "attention_only")
    medians = {name: finite_median(flat, name) for name in names}
    defined_states = {name: int(sum(bool(np.isfinite(row[name])) for row in flat)) for name in names}
    improvement = medians["pure_attention"] - medians["early25"]
    improvement_ci = bootstrap_difference(flat, "pure_attention", "early25", 202608151)
    random_ci = bootstrap_difference(flat, "pure_attention", "matched_random", 202608152)
    tasks = sorted(set(row["task"] for row in flat))
    task_comparisons = {}
    for task in tasks:
        task_rows = [row for row in flat if row["task"] == task]
        left, right = finite_median(task_rows, "pure_attention"), finite_median(task_rows, "matched_random")
        task_comparisons[task] = {"pure_attention_median": left, "matched_random_median": right,
                                  "pure_attention_better": bool(np.isfinite(left) and np.isfinite(right) and left > right)}
    tasks_better = int(sum(item["pure_attention_better"] for item in task_comparisons.values()))
    coverage = float(np.mean([row["k"] >= 1 for row in flat])); median_k = float(np.median([row["k"] for row in flat]))
    norm_ratio = finite_median(flat, "norm_ratio")
    checks = {"pure_attention_cosine_ge_0_65": bool(medians["pure_attention"] >= .65),
              "median_improvement_ge_0_08": bool(improvement >= .08),
              "improvement_ci_lower_gt_zero": bool(improvement_ci[0] > 0),
              "random_difference_ci_lower_gt_zero": bool(random_ci[0] > 0),
              "at_least_8_tasks_better_than_random": bool(tasks_better >= 8),
              "norm_ratio_in_0_6_1_3": bool(.6 <= norm_ratio <= 1.3),
              "coverage_ge_0_90": bool(coverage >= .9),
              "median_selected_count_ge_2": bool(median_k >= 2)}
    passed = bool(all(checks.values()))
    summary = {"status": "GO" if passed else "NO_GO", "states": len(flat), "medians_on_defined_states": medians,
               "metric_defined_states": defined_states, "pure_attention_minus_early25": improvement,
               "improvement_bootstrap_95_ci": improvement_ci,
               "pure_attention_minus_random_bootstrap_95_ci": random_ci,
               "tasks_better_than_random": tasks_better, "task_comparisons": task_comparisons,
               "coverage": coverage, "median_selected_token_count": median_k,
               "median_effect_norm_ratio_on_defined_states": norm_ratio,
               "undefined_cosine_policy": "Exclude no-op k=0 states from direction metrics; enforce them through the locked coverage gate.",
               "checks": checks, "rollout_authorized": passed}
    (artifact / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd-root", type=Path, required=True)
    parser.add_argument("--stage-a", type=Path, required=True)
    parser.add_argument("--early-artifact", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    pcd_root, stage_a, early_artifact, artifact = map(Path.resolve, (args.pcd_root, args.stage_a, args.early_artifact, args.artifact))
    artifact.mkdir(parents=True, exist_ok=True)
    checkpoint = pcd_root / "source/PCD/pretrained/openvla-7b"; sys.path.insert(0, str(pcd_root / "source/PCD"))
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(checkpoint, attn_implementation="eager", torch_dtype=torch.bfloat16,
                                                    low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=True).cuda().eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    means = torch.load(early_artifact / "early_patch_means.pt", map_location="cpu", weights_only=True)["means"]
    if len(means) != 2 or any(value.ndim != 2 or value.shape[0] != 256 for value in means):
        raise RuntimeError("Expected position-conditioned means for two 256-patch vision branches")
    states = read_jsonl(stage_a / "states.lock.jsonl"); states = states[:args.limit] if args.limit else states
    stage_results = {item["state_id"]: item for item in read_jsonl(stage_a / "results.jsonl")}
    results = []
    with (artifact / "results.jsonl").open("w") as handle:
        for ordinal, row in enumerate(states, 1):
            mask = json.loads((stage_a / "masks" / f"{row['state_id']}.json").read_text())
            overlaps = np.asarray(mask["token_overlap_ratio"], dtype=float)
            if overlaps.shape != (256,) or not np.isfinite(overlaps).all() or np.any((overlaps < 0) | (overlaps > 1)):
                raise RuntimeError(f"Invalid patch overlaps for {row['state_id']}")
            early25 = np.flatnonzero(overlaps >= .25).astype(int).tolist()
            pure80 = np.flatnonzero(overlaps >= .80).astype(int).tolist()
            inputs = load_inputs(processor, pcd_root / row["clean_path"], row["instruction"], model)
            clean_ids = torch.tensor([stage_results[row["state_id"]]["clean_action_tokens"]], device=model.device,
                                     dtype=inputs["input_ids"].dtype)
            clean_attention = teacher_forced_forward(model, inputs, clean_ids, record_attention=True).attention_scores.numpy()
            if clean_attention.shape != (256,) or not np.isfinite(clean_attention).all():
                raise RuntimeError(f"Invalid clean attention for {row['state_id']}")
            k = math.ceil(len(pure80) / 2)
            pure_attention = stable_top(clean_attention, pure80, k)
            background = np.flatnonzero(overlaps < .25).astype(int)
            random_seed = int(hashlib.sha256((row["state_id"] + "|PURE80_ATTN_R1").encode()).hexdigest()[:16], 16)
            matched_random = sorted(np.random.default_rng(random_seed).choice(background, size=k, replace=False).astype(int).tolist()) if k else []
            attention_only = stable_top(clean_attention, list(range(256)), k)
            if len(pure_attention) != k or len(set(pure_attention)) != k or not set(pure_attention).issubset(pure80):
                raise RuntimeError(f"Invalid Pure80+Attention selection for {row['state_id']}")
            if len(matched_random) != k or len(set(matched_random)) != k or any(overlaps[i] >= .25 for i in matched_random):
                raise RuntimeError(f"Invalid matched-random selection for {row['state_id']}")
            if len(attention_only) != k or len(set(attention_only)) != k:
                raise RuntimeError(f"Invalid attention-only selection for {row['state_id']}")
            with np.load(stage_a / "logits" / f"{row['state_id']}.npz") as logits_np:
                clean = torch.from_numpy(logits_np["clean"].copy())
                pixel = torch.from_numpy(logits_np["pixel_cf"].copy())
            logits = {"early25": branch(model, inputs, clean_ids, early25, means),
                      "pure80": branch(model, inputs, clean_ids, pure80, means),
                      "pure_attention": branch(model, inputs, clean_ids, pure_attention, means),
                      "matched_random": branch(model, inputs, clean_ids, matched_random, means),
                      "attention_only": branch(model, inputs, clean_ids, attention_only, means)}
            branch_metrics = {name: metrics(model, clean, pixel, value) for name, value in logits.items()}
            pixel_pcd = pcd_ids(clean, pixel)
            detail = {}
            for name, value in logits.items():
                ids = pcd_ids(clean, value)
                detail[name] = {**branch_metrics[name], "pcd_action_tokens": ids.tolist(),
                                "pcd_argmax_flip_positions": (ids != clean.argmax(-1)).nonzero().flatten().tolist(),
                                "pixel_pcd_position_agreement": float((ids == pixel_pcd).float().mean())}
            token_details = [{"token_id": token, "object_overlap": float(overlaps[token]),
                              "clean_attention": float(clean_attention[token]),
                              "is_pure80": token in pure80, "is_pure_attention": token in pure_attention,
                              "is_matched_random": token in matched_random,
                              "is_attention_only": token in attention_only} for token in range(256)]
            result = {"state_id": row["state_id"], "task": row["task"],
                      "sam_mask_area": float(np.mean(overlaps)), "token_count_25": len(early25),
                      "token_count_80": len(pure80), "selected_token_count": k,
                      "pure80_token_ids": pure80, "pure_attention_token_ids": pure_attention,
                      "matched_random_token_ids": matched_random, "attention_only_token_ids": attention_only,
                      "token_details": token_details, "branches": detail,
                      "pixel_pcd_action_tokens": pixel_pcd.tolist()}
            action_slice = action_token_slice(model)
            np.savez_compressed(artifact / f"{row['state_id']}.npz", clean=clean[:, action_slice].numpy(),
                                pixel=pixel[:, action_slice].numpy(),
                                **{name: value[:, action_slice].numpy() for name, value in logits.items()})
            handle.write(json.dumps(result, sort_keys=True) + "\n"); handle.flush(); results.append(result)
            print(json.dumps({"ordinal": ordinal, "total": len(states), "state_id": row["state_id"],
                              "k": k, "pure_attention_cosine": detail["pure_attention"]["cosine"]}), flush=True)
    if args.limit: return
    summary = summarize(results, artifact)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__": main()
