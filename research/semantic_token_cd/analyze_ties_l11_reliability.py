"""Offline test of whether TIES-style rank rigidity predicts L11-SHR Harm.

The primary metric follows TIES conceptually: mean Kendall rank correlation
between adjacent language layers, computed over all 256 visual-token scores.
The statistical unit is an episode (three states are averaged), not a frame.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, mannwhitneyu


REPO = Path(__file__).resolve().parents[2]
ARTIFACT = REPO / "artifacts/ties_l11_reliability_diagnostic_v1"
LAYER_STATES = REPO / "artifacts/prompt_attn_layer_selection_v1/states"
CLOSE_STATES = REPO / "artifacts/prompt_attn_l11_budget_diagnostic_v1/states"
L11_THREE = REPO / "artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes"
L11_CLOSE = REPO / "artifacts/prompt_attn_layer_selection_v1/closed_loop_remaining6/episodes"
VANILLA = REPO / "artifacts/vanilla_recon_shr_canonical_0_299_v2/episodes"

METRICS = {
    "tau_all_adjacent": tuple(range(32)),
    "tau_l7_l15_adjacent": tuple(range(7, 16)),
    "tau_l10_l12_adjacent": tuple(range(10, 13)),
}
CATEGORY_ORDER = ("rescue", "harm", "both_success", "both_fail")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def episode_result(task: str, seed: int) -> tuple[bool, bool]:
    l11_root = L11_CLOSE if task == "google_robot_close_drawer" else L11_THREE
    l11_path = l11_root / task / "prompt_single" / f"episode_{seed:03d}_summary.json"
    vanilla_path = VANILLA / task / "vanilla" / f"episode_{seed:03d}_summary.json"
    if not l11_path.exists() or not vanilla_path.exists():
        raise FileNotFoundError((l11_path, vanilla_path))
    l11, vanilla = read_json(l11_path), read_json(vanilla_path)
    if l11.get("technical_pass") is not True:
        raise RuntimeError(f"L11 technical audit failed: {l11_path}")
    hash_keys = ("canonical_snapshot_sha256", "initial_state_sha256", "initial_rgb_sha256")
    if any(l11.get(key) != vanilla.get(key) for key in hash_keys):
        raise RuntimeError(f"L11/Vanilla pairing failed: {task} seed={seed}")
    return bool(vanilla["success"]), bool(l11["success"])


def category(vanilla: bool, l11: bool) -> str:
    if not vanilla and l11:
        return "rescue"
    if vanilla and not l11:
        return "harm"
    return "both_success" if vanilla else "both_fail"


def adjacent_tau(scores: np.ndarray, layers: tuple[int, ...]) -> float:
    values = []
    for left, right in zip(layers[:-1], layers[1:]):
        value = float(kendalltau(scores[left], scores[right], variant="b").statistic)
        if not np.isfinite(value):
            raise FloatingPointError(f"invalid Kendall tau at layers {left}/{right}")
        values.append(value)
    return float(np.mean(values))


def auc_high_predicts_harm(harm: np.ndarray, rescue: np.ndarray) -> float | None:
    if not len(harm) or not len(rescue):
        return None
    # Probability that a randomly selected Harm episode has greater rigidity.
    wins = sum(float(h > r) + 0.5 * float(h == r) for h in harm for r in rescue)
    return wins / (len(harm) * len(rescue))


def main() -> None:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    paths = sorted(LAYER_STATES.glob("*/*/*.npz")) + sorted(CLOSE_STATES.glob("*/*/*.npz"))
    if len(paths) != 105:
        raise RuntimeError(f"expected 105 cached full-layer states, found {len(paths)}")

    grouped_paths: dict[tuple[str, int], list[Path]] = defaultdict(list)
    for path in paths:
        task = path.parents[1].name
        seed = int(path.parent.name.removeprefix("seed_"))
        grouped_paths[(task, seed)].append(path)

    state_rows = []
    for (task, seed), episode_paths in sorted(grouped_paths.items()):
        vanilla, l11 = episode_result(task, seed)
        ordered = sorted(episode_paths, key=lambda p: int(p.stem.removeprefix("step_")))
        phase_names = ("early", "middle", "late")
        if len(ordered) != 3:
            raise RuntimeError(f"expected three states for {task} seed={seed}")
        for phase, path in zip(phase_names, ordered):
            with np.load(path) as payload:
                scores = np.asarray(payload["original"], dtype=np.float64)
            if scores.shape != (32, 256) or not np.isfinite(scores).all():
                raise RuntimeError(f"invalid attention cache: {path}")
            row = {
                "task": task,
                "seed": seed,
                "step": int(path.stem.removeprefix("step_")),
                "phase": phase,
                "category": category(vanilla, l11),
                "vanilla_success": vanilla,
                "l11_success": l11,
                "source": str(path),
            }
            for name, layers in METRICS.items():
                row[name] = adjacent_tau(scores, layers)
            row["tau_l11_to_all"] = float(np.mean([
                kendalltau(scores[11], scores[layer], variant="b").statistic
                for layer in range(32) if layer != 11
            ]))
            state_rows.append(row)

    episode_rows = []
    for (task, seed), rows in sorted(defaultdict(list, {
        key: [x for x in state_rows if (x["task"], x["seed"]) == key]
        for key in grouped_paths
    }).items()):
        episode_rows.append({
            "task": task,
            "seed": seed,
            "category": rows[0]["category"],
            "vanilla_success": rows[0]["vanilla_success"],
            "l11_success": rows[0]["l11_success"],
            **{name: float(np.mean([row[name] for row in rows]))
               for name in (*METRICS.keys(), "tau_l11_to_all")},
        })

    fieldnames = list(state_rows[0])
    with (ARTIFACT / "state_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(state_rows)
    with (ARTIFACT / "episode_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
        writer.writeheader(); writer.writerows(episode_rows)

    summaries = {}
    for scope in ["overall", *sorted({row["task"] for row in episode_rows})]:
        subset = episode_rows if scope == "overall" else [x for x in episode_rows if x["task"] == scope]
        summaries[scope] = {}
        for label in CATEGORY_ORDER:
            selected = [x for x in subset if x["category"] == label]
            summaries[scope][label] = {
                "episodes": len(selected),
                **{name: (float(np.mean([x[name] for x in selected])) if selected else None)
                   for name in (*METRICS.keys(), "tau_l11_to_all")},
            }

    tests = {}
    for name in (*METRICS.keys(), "tau_l11_to_all"):
        harm = np.asarray([x[name] for x in episode_rows if x["category"] == "harm"])
        rescue = np.asarray([x[name] for x in episode_rows if x["category"] == "rescue"])
        test = mannwhitneyu(harm, rescue, alternative="greater", method="exact")
        tests[name] = {
            "hypothesis": "Harm rigidity > Rescue rigidity",
            "harm_n": len(harm), "rescue_n": len(rescue),
            "harm_mean": float(harm.mean()), "rescue_mean": float(rescue.mean()),
            "mean_difference": float(harm.mean() - rescue.mean()),
            "mann_whitney_u": float(test.statistic), "one_sided_exact_p": float(test.pvalue),
            "auc_high_tau_predicts_harm": auc_high_predicts_harm(harm, rescue),
        }

    primary = tests["tau_all_adjacent"]
    task_directions = sum(
        1 for task, values in summaries.items() if task != "overall"
        and values["harm"]["episodes"] and values["rescue"]["episodes"]
        and values["harm"]["tau_all_adjacent"] > values["rescue"]["tau_all_adjacent"]
    )
    eligible_tasks = sum(
        1 for task, values in summaries.items() if task != "overall"
        and values["harm"]["episodes"] and values["rescue"]["episodes"]
    )
    supports = bool(
        primary["one_sided_exact_p"] < 0.05
        and primary["auc_high_tau_predicts_harm"] >= 0.70
        and task_directions >= min(3, eligible_tasks)
    )
    result = {
        "protocol": "TIES_L11_RELIABILITY_DIAGNOSTIC_V1",
        "primary_metric": "mean adjacent-layer Kendall tau over layers 0-31",
        "statistical_unit": "episode mean over early/middle/late states",
        "states": len(state_rows), "episodes": len(episode_rows),
        "summaries": summaries, "tests": tests,
        "primary_decision": {
            "supports_ties_gate": supports,
            "tasks_with_harm_greater_than_rescue": task_directions,
            "eligible_tasks": eligible_tasks,
            "rule": "p<.05, AUC>=.70, and same direction in at least 3 eligible tasks",
            "guardrail": "exploratory outcome-selected cache; any threshold needs separate validation",
        },
    }
    (ARTIFACT / "FINAL_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, name in zip(axes, METRICS):
        groups = [[x[name] for x in episode_rows if x["category"] == label] for label in CATEGORY_ORDER]
        ax.boxplot(groups, tick_labels=[x.replace("_", "\n") for x in CATEGORY_ORDER], showmeans=True)
        ax.set_title(name); ax.set_ylabel("mean Kendall tau")
    fig.tight_layout(); fig.savefig(ARTIFACT / "tau_by_outcome.png", dpi=180); plt.close(fig)

    lines = [
        "# TIES-style L11 reliability diagnostic", "",
        f"- Cached full-layer states: {len(state_rows)} from {len(episode_rows)} episodes.",
        "- Primary unit: episode mean across early/middle/late states (no frame pseudo-replication).",
        "- Primary hypothesis: L11 Harm episodes have higher inter-layer rank rigidity than Rescue episodes.", "",
        "## Primary result", "",
        "| Metric | Harm | Rescue | Difference | AUC | one-sided exact p |",
        "|---|---:|---:|---:|---:|---:|",
        f"| all adjacent layers | {primary['harm_mean']:.4f} | {primary['rescue_mean']:.4f} | "
        f"{primary['mean_difference']:+.4f} | {primary['auc_high_tau_predicts_harm']:.3f} | "
        f"{primary['one_sided_exact_p']:.4g} |", "",
        f"Decision: **{'supports' if supports else 'does not support'}** using TIES rigidity as a Harm gate under the locked exploratory rule.", "",
        "## Per-task primary metric", "",
        "| Task | Rescue n/mean | Harm n/mean | Harm-Rescue |", "|---|---:|---:|---:|",
    ]
    for task in sorted(x for x in summaries if x != "overall"):
        values = summaries[task]; rescue = values["rescue"]; harm = values["harm"]
        difference = None if not rescue["episodes"] or not harm["episodes"] else harm["tau_all_adjacent"] - rescue["tau_all_adjacent"]
        lines.append(
            f"| {task.removeprefix('google_robot_')} | {rescue['episodes']}/{rescue['tau_all_adjacent'] if rescue['episodes'] else float('nan'):.4f} | "
            f"{harm['episodes']}/{harm['tau_all_adjacent'] if harm['episodes'] else float('nan'):.4f} | "
            f"{difference if difference is not None else float('nan'):+.4f} |"
        )
    lines += ["", "## Guardrail", "",
              "These cached episodes were selected for earlier layer diagnostics and are not a random confirmatory sample. "
              "A positive signal justifies a separately locked threshold test; it does not validate a closed-loop gate by itself."]
    (ARTIFACT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"primary": primary, "decision": result["primary_decision"]}, indent=2))


if __name__ == "__main__":
    main()
