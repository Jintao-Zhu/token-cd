"""Offline Positive-Support Constrained SHR (PSC-SHR) analysis.

The canonical release stores positive/negative action-vocabulary logits as
float16 arrays with shape [control_step, 7, 256].  This program reconstructs
lambda=0.5 SHR, applies positive Top-K and relative-probability support masks,
and writes compact sparse representations of every filtered logit vector.

Offline action changes are exact with respect to the stored logits.  They are
not counterfactual rollout outcomes: success/Rescue/Harm for PSC requires a new
closed-loop rollout whenever any selected action token changes.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


TASKS = (
    "google_robot_close_drawer", "google_robot_open_drawer",
    "google_robot_pick_coke_can", "google_robot_move_near",
)
TOP_KS = (10, 20, 50)
LAMBDA = 0.5
RELATIVE_THRESHOLD = 0.1
ACTION_BINS = 256
ACTION_VOCAB_START = 31744  # model.vocab_size(32000) - 256 action bins
GUIDED_DIMS = 6


def atomic_json(path: Path, payload) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def topk_sparse(positive: np.ndarray, fused: np.ndarray, k: int):
    indices = np.argpartition(positive, -k, axis=-1)[..., -k:]
    scores = np.take_along_axis(positive, indices, axis=-1)
    order = np.argsort(-scores, axis=-1, kind="stable")
    indices = np.take_along_axis(indices, order, axis=-1)
    values = np.take_along_axis(fused, indices, axis=-1)
    # Match dense numpy/torch argmax: equal maxima select the smallest original
    # vocabulary index, not whichever candidate happens to appear first after
    # sorting by positive support.
    maxima = values.max(axis=-1, keepdims=True)
    winner = np.where(values == maxima, indices, ACTION_BINS).min(axis=-1)
    return indices.astype(np.uint8), values.astype(np.float16), winner.astype(np.uint8)


def category(vanilla: bool, shr: bool) -> str:
    if not vanilla and shr:
        return "shr_rescue"
    if vanilla and not shr:
        return "shr_harm"
    return "both_success" if vanilla else "both_fail"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    canonical, output = args.canonical.resolve(), args.output.resolve()
    derived = output / "episode_logits_sparse"
    derived.mkdir(parents=True, exist_ok=True)
    rows_path = output / "action_token_analysis.csv.gz"
    stats = {task: {"dimensions": 0, "guided_dimensions": 0, "episodes": 0,
                    "baseline_fp16_mismatch": 0, "technical_pass": 0,
                    "methods": {}} for task in TASKS}
    outcome_stats = {task: defaultdict(lambda: defaultdict(lambda: {
        "episodes": 0, "episodes_changed": 0, "dimensions": 0, "dimensions_changed": 0
    })) for task in TASKS}
    fieldnames = [
        "task", "seed", "step", "dimension", "outcome_group",
        "argmax_positive", "positive_token_id_exact", "argmax_shr_exact",
        "shr_token_id_exact", "argmax_shr_offline",
        "positive_rank_of_shr", "baseline_fp16_reproduced",
        "argmax_psc_k10", "changed_k10", "argmax_psc_k20", "changed_k20",
        "argmax_psc_k50", "changed_k50", "argmax_psc_rel010", "changed_rel010",
    ]
    with gzip.open(rows_path, "wt", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
        writer.writeheader()
        for task in TASKS:
            task_source = canonical / "episodes" / task
            task_output = derived / task
            task_output.mkdir(exist_ok=True)
            for seed in range(300):
                arm = task_source / "shr_harmonic"
                arrays_path = arm / f"episode_{seed:03d}_arrays.npz"
                summary_path = arm / f"episode_{seed:03d}_summary.json"
                vanilla_path = task_source / "vanilla" / f"episode_{seed:03d}_summary.json"
                with np.load(arrays_path) as source:
                    positive = source["positive_logits"].astype(np.float32)
                    negative = source["negative_logits"].astype(np.float32)
                if positive.shape != negative.shape or positive.shape[1:] != (7, ACTION_BINS):
                    raise RuntimeError(f"bad logits shape: {arrays_path}: {positive.shape}/{negative.shape}")
                summary = json.loads(summary_path.read_text())
                vanilla = json.loads(vanilla_path.read_text())
                trace = summary["selector_trace"]
                if len(trace) != positive.shape[0]:
                    raise RuntimeError(f"trace/logit length mismatch: {summary_path}")
                positive_ids = np.asarray([x["positive_token_ids"] for x in trace], dtype=np.int64)
                exact_ids = np.asarray([x["final_token_ids"] for x in trace], dtype=np.int64)
                positive_argmax = positive.argmax(axis=-1)
                exact = exact_ids - ACTION_VOCAB_START
                # OpenVLA may use EOS token 2 as the seventh/gripper output. SHR
                # deliberately leaves q=6 untouched, so represent such a token
                # with action-bin sentinel -1 and pass it through for every PSC arm.
                invalid = (exact < 0) | (exact >= ACTION_BINS)
                if np.any(invalid[:, :GUIDED_DIMS]):
                    raise RuntimeError(f"guided final token outside action vocabulary: {summary_path}")
                exact[invalid] = -1
                fused = (1.0 + LAMBDA) * positive - LAMBDA * negative
                fused[:, 6] = positive[:, 6]
                offline_shr = fused.argmax(axis=-1)
                reproduced = offline_shr == exact
                reproduced[:, 6] = True  # q=6 is an exact full-vocabulary passthrough
                # Best rank under ties: one plus number of strictly larger logits.
                rank_indices = np.maximum(exact, 0)
                exact_positive = np.take_along_axis(positive, rank_indices[..., None], axis=-1)[..., 0]
                positive_rank = 1 + (positive > exact_positive[..., None]).sum(axis=-1)
                positive_rank[invalid] = 0
                winners = {}
                sparse = {
                    "fused_logits": fused.astype(np.float16),
                    "positive_argmax": positive_argmax.astype(np.uint8),
                    "positive_token_ids_exact": positive_ids.astype(np.int32),
                    "shr_token_ids_exact": exact_ids.astype(np.int32),
                    "shr_argmax_exact": exact.astype(np.int16),
                    "shr_argmax_offline": offline_shr.astype(np.uint8),
                    "positive_rank_of_shr": positive_rank.astype(np.uint16),
                    "source_arrays_path": np.asarray(str(arrays_path.relative_to(canonical))),
                }
                for k in TOP_KS:
                    indices, values, winner = topk_sparse(positive, fused, k)
                    winner = winner.astype(np.int16)
                    winner[:, 6] = exact[:, 6]
                    winners[f"k{k}"] = winner
                    sparse[f"psc_k{k}_indices"] = indices
                    sparse[f"psc_k{k}_fused_logits"] = values
                rel_mask = positive > (positive.max(axis=-1, keepdims=True) + math.log(RELATIVE_THRESHOLD))
                rel_filtered = np.where(rel_mask, fused, -np.inf)
                winners["rel010"] = rel_filtered.argmax(axis=-1).astype(np.uint8)
                winners["rel010"] = winners["rel010"].astype(np.int16)
                winners["rel010"][:, 6] = exact[:, 6]
                sparse["psc_rel010_mask_packbits"] = np.packbits(rel_mask, axis=-1)
                sparse["psc_rel010_allowed_count"] = rel_mask.sum(axis=-1).astype(np.uint16)
                np.savez_compressed(task_output / f"episode_{seed:03d}_psc_logits.npz", **sparse)

                group = category(bool(vanilla["success"]), bool(summary["success"]))
                stat = stats[task]
                stat["episodes"] += 1
                stat["dimensions"] += int(exact.size)
                stat["guided_dimensions"] += int(exact[:, :GUIDED_DIMS].size)
                stat["baseline_fp16_mismatch"] += int((~reproduced[:, :GUIDED_DIMS]).sum())
                stat["technical_pass"] += int(bool(summary.get("technical_pass")))
                for method, winner in winners.items():
                    changed = winner != exact
                    m = stat["methods"].setdefault(method, {
                        "dimensions_changed": 0, "guided_dimensions_changed": 0,
                        "episodes_changed": 0, "changed_by_dimension": [0] * 7,
                        "selected_edge10": 0, "shr_edge10": 0,
                        "allowed_candidates_sum": 0,
                    })
                    m["dimensions_changed"] += int(changed.sum())
                    m["guided_dimensions_changed"] += int(changed[:, :GUIDED_DIMS].sum())
                    m["episodes_changed"] += int(changed.any())
                    for dim in range(7):
                        m["changed_by_dimension"][dim] += int(changed[:, dim].sum())
                    m["selected_edge10"] += int(((winner < 10) | (winner >= 246)).sum())
                    m["shr_edge10"] += int(((exact < 10) | (exact >= 246)).sum())
                    allowed = int(method[1:]) if method.startswith("k") else rel_mask.sum(axis=-1)
                    m["allowed_candidates_sum"] += (int(allowed * exact[:, :GUIDED_DIMS].size)
                                                    if np.isscalar(allowed)
                                                    else int(allowed[:, :GUIDED_DIMS].sum()))
                    o = outcome_stats[task][method][group]
                    o["episodes"] += 1
                    o["episodes_changed"] += int(changed.any())
                    o["dimensions"] += int(changed.size)
                    o["dimensions_changed"] += int(changed.sum())
                for step in range(exact.shape[0]):
                    for dim in range(7):
                        row = {
                            "task": task, "seed": seed, "step": step, "dimension": dim,
                            "outcome_group": group,
                            "argmax_positive": int(positive_argmax[step, dim]),
                            "positive_token_id_exact": int(positive_ids[step, dim]),
                            "argmax_shr_exact": int(exact[step, dim]),
                            "shr_token_id_exact": int(exact_ids[step, dim]),
                            "argmax_shr_offline": int(offline_shr[step, dim]),
                            "positive_rank_of_shr": int(positive_rank[step, dim]),
                            "baseline_fp16_reproduced": int(reproduced[step, dim]),
                        }
                        for method in ("k10", "k20", "k50", "rel010"):
                            row[f"argmax_psc_{method}"] = int(winners[method][step, dim])
                            row[f"changed_{method}"] = int(winners[method][step, dim] != exact[step, dim])
                        writer.writerow(row)
            print(json.dumps({"task_complete": task, "episodes": 300}), flush=True)

    for task in TASKS:
        stat = stats[task]
        for method, m in stat["methods"].items():
            m["dimension_change_rate"] = m["dimensions_changed"] / stat["dimensions"]
            m["guided_dimension_change_rate"] = m["guided_dimensions_changed"] / stat["guided_dimensions"]
            m["episode_change_rate"] = m["episodes_changed"] / stat["episodes"]
            m["mean_allowed_candidates"] = m["allowed_candidates_sum"] / stat["guided_dimensions"]
            m["edge10_delta"] = m["selected_edge10"] - m["shr_edge10"]
        stat["baseline_fp16_mismatch_rate"] = stat["baseline_fp16_mismatch"] / stat["guided_dimensions"]
        stat["outcome_strata"] = {method: dict(groups) for method, groups in outcome_stats[task].items()}
    payload = {
        "protocol": "PSC_SHR_OFFLINE_V1", "complete": True, "tasks": list(TASKS),
        "episodes": 1200, "lambda": LAMBDA, "top_k": list(TOP_KS),
        "relative_probability_threshold": RELATIVE_THRESHOLD,
        "action_vocab_start": ACTION_VOCAB_START,
        "psc_dimensions": "0..5; dimension 6/gripper is exact positive passthrough",
        "storage": {
            "positive_negative": "referenced from canonical SHR arrays (float16 [step,7,256])",
            "fused": "full float16 array in each derived episode NPZ",
            "filtered_topk": "sparse indices+fused logits; all unlisted logits are -inf",
            "filtered_relative": "packbits mask over fused logits; false entries are -inf",
        },
        "limitation": "Offline token changes are not closed-loop PSC success outcomes; changed actions require rollout.",
        "by_task": stats,
    }
    atomic_json(output / "summary.json", payload)
    lines = [
        "# Positive-Support Constrained SHR — offline report", "",
        "PSC is evaluated on stored float16 action-vocabulary logits. A changed token is not a",
        "counterfactual success label; PSC Rescue/Harm requires closed-loop rollout.", "",
        "| Task | Method | Changed dims | Change rate | Changed episodes | Episode rate | Mean support |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        for method, m in stats[task]["methods"].items():
            lines.append(f"| {task.replace('google_robot_', '')} | {method} | {m['dimensions_changed']} | "
                         f"{m['dimension_change_rate']:.4%} | {m['episodes_changed']} | "
                         f"{m['episode_change_rate']:.2%} | {m['mean_allowed_candidates']:.2f} |")
    lines.extend(["", "## Interpretation guardrail", "",
                  "Existing Vanilla/SHR outcomes are used only to stratify which old trajectories are touched.",
                  "They are not relabeled as PSC outcomes. Run paired PSC rollouts to measure Rescue/Harm."])
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"complete": True, "episodes": 1200, "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
