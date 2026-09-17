"""Offline discovery audit for conservative L11 risk-gated lambda."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from research.semantic_token_cd.prompt_attn_l11_risk_gated_policy import (
    LOW_LAMBDA,
    PROBE_LAMBDA,
    RiskSignalTracker,
    RiskThresholds,
    gate_decision,
    soft_normalized_actions,
)
from research.semantic_token_cd.prompt_attn_shr_rollout import atomic_json


TASKS = (
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
    "widowx_carrot_on_plate",
)
SHORT = {
    "google_robot_close_drawer": "close_drawer",
    "google_robot_pick_coke_can": "pick_coke_can",
    "google_robot_move_near": "move_near",
    "widowx_carrot_on_plate": "carrot_on_plate",
}
SOURCE_BY_TASK = {
    "google_robot_close_drawer": "prompt_attn_layer_selection_v1/closed_loop_remaining6",
    "google_robot_pick_coke_can": "prompt_attn_layer_selection_v1/closed_loop",
    "google_robot_move_near": "prompt_attn_layer_selection_v1/closed_loop",
    "widowx_carrot_on_plate": "prompt_attn_layer_selection_v1/closed_loop_remaining6",
}
BASE_COSINES = (0.5, 0.7)
GUIDE_COSINES = (-0.2, 0.0)
MAGNITUDE_RATIOS = (1.25, 1.5)
BOOTSTRAP_SAMPLES = 2000


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def classify(vanilla: bool, l11: bool) -> str:
    if vanilla and not l11:
        return "fixed_cd_harm"
    if not vanilla and l11:
        return "rescue"
    if vanilla and l11:
        return "stable_success"
    return "stable_fail"


def reconstruct_trace(arrays_path: Path) -> list[dict]:
    arrays = np.load(arrays_path)
    positive_logits = arrays["positive"][:, :6].astype(np.float64)
    negative_logits = arrays["negative"][:, :6].astype(np.float64)
    probe_logits = (1.0 + PROBE_LAMBDA) * positive_logits - PROBE_LAMBDA * negative_logits
    positive_actions = soft_normalized_actions(positive_logits)
    probe_actions = soft_normalized_actions(probe_logits)
    tracker = RiskSignalTracker()
    return [tracker.observe(positive, probe) for positive, probe in zip(positive_actions, probe_actions)]


def threshold_name(thresholds: RiskThresholds) -> str:
    return f"b{thresholds.base_cosine:g}_g{thresholds.guide_cosine:g}_m{thresholds.magnitude_ratio:g}"


def threshold_metrics(episodes: list[dict], thresholds: RiskThresholds) -> dict:
    episode_rows = []
    total_triggers = 0
    total_steps = 0
    for episode in episodes:
        triggers = [gate_decision(step, thresholds) for step in episode["trace"]]
        count = sum(triggers)
        total_triggers += count
        total_steps += len(triggers)
        episode_rows.append({
            **episode,
            "gate_hit": count > 0,
            "trigger_count": count,
            "trigger_fraction": count / len(triggers),
        })

    def mean_for(category_filter, key: str) -> float | None:
        selected = [row[key] for row in episode_rows if category_filter(row)]
        return float(np.mean(selected)) if selected else None

    harm_hit = mean_for(lambda row: row["category"] == "fixed_cd_harm", "gate_hit")
    success_hit = mean_for(lambda row: row["l11"], "gate_hit")
    rescue_hit = mean_for(lambda row: row["category"] == "rescue", "gate_hit")
    stable_success_hit = mean_for(lambda row: row["category"] == "stable_success", "gate_hit")
    by_task = {}
    positive_enrichment_tasks = 0
    for task in TASKS:
        task_harm = mean_for(
            lambda row, task=task: row["task"] == task and row["category"] == "fixed_cd_harm",
            "gate_hit",
        )
        task_success = mean_for(
            lambda row, task=task: row["task"] == task and row["l11"], "gate_hit"
        )
        enrichment = task_harm - task_success if task_harm is not None and task_success is not None else None
        if enrichment is not None and enrichment > 0:
            positive_enrichment_tasks += 1
        by_task[SHORT[task]] = {
            "harm_hit_rate": task_harm,
            "l11_success_hit_rate": task_success,
            "enrichment": enrichment,
        }
    trigger_rate = total_triggers / total_steps
    return {
        "threshold_id": threshold_name(thresholds),
        "base_cosine": thresholds.base_cosine,
        "guide_cosine": thresholds.guide_cosine,
        "magnitude_ratio": thresholds.magnitude_ratio,
        "harm_episode_hit_rate": harm_hit,
        "l11_success_episode_hit_rate": success_hit,
        "rescue_episode_hit_rate": rescue_hit,
        "stable_success_episode_hit_rate": stable_success_hit,
        "episode_hit_enrichment": harm_hit - success_hit,
        "overall_timestep_trigger_rate": trigger_rate,
        "implied_mean_lambda": PROBE_LAMBDA - (PROBE_LAMBDA - LOW_LAMBDA) * trigger_rate,
        "positive_enrichment_task_count": positive_enrichment_tasks,
        "trigger_rate_constraint_pass": trigger_rate <= 0.20,
        "by_task": by_task,
    }


def select_threshold(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["trigger_rate_constraint_pass"]]
    if not valid:
        raise RuntimeError("no threshold satisfies the 20% trigger-rate constraint")
    return max(valid, key=lambda row: (
        row["episode_hit_enrichment"],
        -row["stable_success_episode_hit_rate"],
        -row["rescue_episode_hit_rate"],
        -row["overall_timestep_trigger_rate"],
        row["base_cosine"],
        -row["guide_cosine"],
        row["magnitude_ratio"],
    ))


def bootstrap_auc(labels: np.ndarray, scores: np.ndarray) -> dict:
    rng = np.random.default_rng(20260913)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    values = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = np.concatenate([
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        ])
        values.append(float(roc_auc_score(labels[sample], scores[sample])))
    return {
        "samples": BOOTSTRAP_SAMPLES,
        "p2_5": percentile(values, 2.5),
        "median": percentile(values, 50),
        "p97_5": percentile(values, 97.5),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=49)
    args = parser.parse_args()
    repo = args.repo.resolve()
    artifact = args.artifact.resolve()
    baseline_csv = repo / "artifacts/prompt_attn_l11_matched_full_9task_0_299_v1/paired_results_0_299.csv"
    baselines = {}
    with baseline_csv.open() as handle:
        for row in csv.DictReader(handle):
            baselines[(row["task"], int(row["seed"]))] = {
                "vanilla": row["vanilla"] == "True",
                "shr": row["shr"] == "True",
                "l11": row["l11_matched"] == "True",
            }

    episodes = []
    step_rows = []
    for task in TASKS:
        source = repo / "artifacts" / SOURCE_BY_TASK[task] / "episodes" / task / "prompt_single"
        for seed in range(args.seed_start, args.seed_end + 1):
            summary_path = source / f"episode_{seed:03d}_summary.json"
            arrays_path = source / f"episode_{seed:03d}_arrays.npz"
            if not summary_path.exists() or not arrays_path.exists():
                raise RuntimeError(f"missing fixed L11 trace: {task} seed={seed}")
            summary = json.loads(summary_path.read_text())
            if summary.get("technical_pass") is not True:
                raise RuntimeError(f"fixed L11 technical audit failed: {summary_path}")
            baseline = baselines[(task, seed)]
            if bool(summary["success"]) != baseline["l11"]:
                raise RuntimeError(f"fixed L11 success mismatch: {task} seed={seed}")
            trace = reconstruct_trace(arrays_path)
            risks = [step["risk_score"] for step in trace]
            category = classify(baseline["vanilla"], baseline["l11"])
            episode = {
                "task": task,
                "task_short": SHORT[task],
                "seed": seed,
                **baseline,
                "category": category,
                "control_steps": len(trace),
                "risk_max": max(risks),
                "risk_p90": percentile(risks, 90),
                "trace": trace,
            }
            episodes.append(episode)
            for step in trace:
                step_rows.append({
                    "task": SHORT[task],
                    "seed": seed,
                    "category": category,
                    "l11_success": baseline["l11"],
                    **{key: step[key] for key in (
                        "step_index", "base_cosine", "guide_cosine", "guidance_norm",
                        "guidance_magnitude_ema", "guidance_magnitude_ratio", "risk_score", "signals_valid",
                    )},
                })

    auc_episodes = [episode for episode in episodes if episode["category"] == "fixed_cd_harm" or episode["l11"]]
    labels = np.asarray([episode["category"] == "fixed_cd_harm" for episode in auc_episodes], dtype=int)
    scores = np.asarray([episode["risk_p90"] for episode in auc_episodes], dtype=float)
    auc = float(roc_auc_score(labels, scores))
    auc_bootstrap = bootstrap_auc(labels, scores)

    thresholds = [
        RiskThresholds(base, guide, magnitude)
        for base, guide, magnitude in itertools.product(BASE_COSINES, GUIDE_COSINES, MAGNITUDE_RATIOS)
    ]
    threshold_rows = [threshold_metrics(episodes, threshold) for threshold in thresholds]
    selected = select_threshold(threshold_rows)
    leave_one_task_out = {}
    for held_out in TASKS:
        training = [episode for episode in episodes if episode["task"] != held_out]
        leave_one_task_out[SHORT[held_out]] = select_threshold([
            threshold_metrics(training, threshold) for threshold in thresholds
        ])["threshold_id"]
    selected_threshold_stability = Counter(leave_one_task_out.values())
    go_no_go = {
        "auc_at_least_0_65": auc >= 0.65,
        "harm_enrichment_positive": selected["episode_hit_enrichment"] > 0,
        "trigger_rate_at_most_0_20": selected["overall_timestep_trigger_rate"] <= 0.20,
        "positive_enrichment_on_at_least_two_tasks": selected["positive_enrichment_task_count"] >= 2,
    }
    go_no_go["passed"] = all(go_no_go.values())

    public_episodes = [{key: value for key, value in episode.items() if key != "trace"} for episode in episodes]
    write_csv(artifact / "offline/episode_risk_metrics.csv", public_episodes)
    write_csv(artifact / "offline/per_step_risk_signals.csv", step_rows)
    flat_threshold_rows = [{key: value for key, value in row.items() if key != "by_task"} for row in threshold_rows]
    write_csv(artifact / "offline/threshold_grid.csv", flat_threshold_rows)
    payload = {
        "protocol_id": "PROMPT_ATTN_L11_RISK_GATED_4TASK_V1",
        "discovery_seeds": [args.seed_start, args.seed_end],
        "tasks": list(TASKS),
        "episode_count": len(episodes),
        "category_counts": dict(Counter(episode["category"] for episode in episodes)),
        "risk_auc_harm_vs_l11_success": auc,
        "risk_auc_bootstrap": auc_bootstrap,
        "selected_threshold": selected,
        "leave_one_task_out_selected_thresholds": leave_one_task_out,
        "leave_one_task_out_threshold_frequency": dict(selected_threshold_stability),
        "go_no_go": go_no_go,
        "threshold_grid": threshold_rows,
    }
    atomic_json(artifact / "offline/OFFLINE_RISK_AUDIT.json", payload)

    colors = {
        "fixed_cd_harm": "tab:red", "rescue": "tab:green",
        "stable_success": "tab:blue", "stable_fail": "tab:gray",
    }
    figure, axis = plt.subplots(figsize=(7.5, 6.0))
    for category, color in colors.items():
        selected_steps = [row for row in step_rows if row["category"] == category and row["signals_valid"]]
        axis.scatter(
            [row["base_cosine"] for row in selected_steps],
            [row["guide_cosine"] for row in selected_steps],
            s=[8 + 18 * min(float(row["guidance_magnitude_ratio"]), 4.0) for row in selected_steps],
            alpha=0.18, color=color, label=category,
        )
    axis.axvline(selected["base_cosine"], color="black", ls="--", lw=1)
    axis.axhline(selected["guide_cosine"], color="black", ls="--", lw=1)
    axis.set(xlabel="Base action cosine", ylabel="Guidance cosine", xlim=(-1.02, 1.02), ylim=(-1.02, 1.02))
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(artifact / "offline/base_vs_guide_risk_scatter.png", dpi=180)
    plt.close(figure)

    lines = [
        "# L11 Risk-Gated Lambda — Offline Discovery Audit",
        "",
        f"Discovery set: four tasks, seeds {args.seed_start}--{args.seed_end}, {len(episodes)} fixed-L11 episodes.",
        "",
        f"- Category counts: `{payload['category_counts']}`",
        f"- Harm-vs-L11-success R90 AUC: **{auc:.4f}**",
        f"- Bootstrap 95% interval: **[{auc_bootstrap['p2_5']:.4f}, {auc_bootstrap['p97_5']:.4f}]**",
        f"- Selected threshold: `{selected['threshold_id']}`",
        f"- Harm hit rate: **{selected['harm_episode_hit_rate']:.1%}**",
        f"- L11-success hit rate: **{selected['l11_success_episode_hit_rate']:.1%}**",
        f"- Stable-success hit rate: **{selected['stable_success_episode_hit_rate']:.1%}**",
        f"- Timestep trigger rate: **{selected['overall_timestep_trigger_rate']:.1%}**",
        f"- Implied mean lambda: **{selected['implied_mean_lambda']:.4f}**",
        f"- Go/no-go: **{'PASS' if go_no_go['passed'] else 'STOP'}**",
        "",
        "## Leave-one-task-out threshold selections",
        "",
    ]
    lines.extend(f"- {task}: `{threshold}`" for task, threshold in leave_one_task_out.items())
    (artifact / "offline/OFFLINE_RISK_AUDIT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
