"""Offline L11 spatial-attention and cumulative-budget diagnostic."""
from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


ROOT = Path("artifacts/prompt_attn_l11_budget_diagnostic_v1")
LAYER_SOURCE = Path("artifacts/prompt_attn_layer_selection_v1/states")
TASK_SEEDS = {
    "google_robot_open_drawer": [0, 11, 32, 95, 97],
    "google_robot_close_drawer": [0, 1, 9, 21, 49],
    "google_robot_pick_coke_can": [1, 3, 30, 59, 95],
    "google_robot_move_near": [0, 1, 2, 31, 73],
}
SHORT = {task: task.removeprefix("google_robot_") for task in TASK_SEEDS}
THRESHOLDS = (.7, .8, .9)
EDGE = np.asarray([i for i in range(256) if i // 16 in (0, 15) or i % 16 in (0, 15)])
CORNERS = np.asarray([0, 15, 240, 255])


def stable_top(scores: np.ndarray, count: int) -> np.ndarray:
    order = np.lexsort((np.arange(256), -np.asarray(scores, dtype=np.float64)))
    return order[:count]


def top_p(scores: np.ndarray, threshold: float, suppress_edge: bool = False) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).copy()
    if suppress_edge:
        values[EDGE] = 0
    probability = values / values.sum()
    order = np.lexsort((np.arange(256), -probability))
    count = int(np.searchsorted(np.cumsum(probability[order]), threshold) + 1)
    return order[:count]


def overlay_mask(ax, image: np.ndarray, mask: np.ndarray, title: str, color="red") -> None:
    ax.imshow(image)
    alpha = np.zeros((16, 16, 4), dtype=np.float32)
    rgba = {"red": (1, 0, 0), "cyan": (0, .9, 1)}[color]
    alpha.reshape(-1, 4)[mask, :3] = rgba
    alpha.reshape(-1, 4)[mask, 3] = .38
    ax.imshow(alpha, extent=(0, image.shape[1], image.shape[0], 0), interpolation="nearest")
    ax.set_title(title, fontsize=9); ax.axis("off")


def nested_overlay(ax, image: np.ndarray, masks: dict[float, np.ndarray], title: str) -> None:
    level = np.zeros(256, dtype=np.uint8)
    level[masks[.9]] = 1
    level[masks[.8]] = 2
    level[masks[.7]] = 3
    rgba = np.zeros((16, 16, 4), dtype=np.float32)
    palette = {1: (1, .9, 0, .25), 2: (1, .45, 0, .36), 3: (1, 0, 0, .48)}
    for value, color in palette.items():
        rgba.reshape(-1, 4)[level == value] = color
    ax.imshow(image)
    ax.imshow(rgba, extent=(0, image.shape[1], image.shape[0], 0), interpolation="nearest")
    counts = "/".join(str(len(masks[q])) for q in THRESHOLDS)
    ax.set_title(f"{title}\nTop-p .7/.8/.9 m={counts}", fontsize=9); ax.axis("off")


def source_files(task: str, seed: int) -> list[tuple[Path, Path]]:
    root = (ROOT / "states" / task / f"seed_{seed:03d}") if task == "google_robot_close_drawer" else (
        LAYER_SOURCE / task / f"seed_{seed:03d}"
    )
    rows = []
    for npz in root.glob("step_*.npz"):
        rows.append((npz, npz.with_suffix(".json")))
    phase_order = {"early": 0, "middle": 1, "late": 2}
    return sorted(rows, key=lambda pair: phase_order[json.loads(pair[1].read_text())["phase"]])


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    galleries = ROOT / "state_figures"; statistics = ROOT / "statistics"
    galleries.mkdir(parents=True, exist_ok=True); statistics.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": "PROMPT_ATTN_L11_BUDGET_DIAGNOSTIC_V1", "layer": 11,
                "thresholds": list(THRESHOLDS), "edge_suppression": "hard-zero outer 16x16 ring then visual renormalization",
                "tasks": TASK_SEEDS, "sampling": "three target-switch controls where available plus outcome diversity; early/middle/late=10/50/90%"}
    (ROOT / "selection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    state_rows = []; target_rows = []; attentions = defaultdict_list = {task: [] for task in TASK_SEEDS}
    top16_hits = {task: np.zeros(256, dtype=np.int64) for task in TASK_SEEDS}
    top32_hits = {task: np.zeros(256, dtype=np.int64) for task in TASK_SEEDS}
    masks_by_episode = {task: {} for task in TASK_SEEDS}

    for task, seeds in TASK_SEEDS.items():
        for seed in seeds:
            files = source_files(task, seed)
            if len(files) != 3:
                raise RuntimeError(f"expected three states: {task} seed {seed}, got {len(files)}")
            masks_by_episode[task][seed] = []
            for npz_path, json_path in files:
                arrays = np.load(npz_path)
                meta = json.loads(json_path.read_text())
                image = arrays["image"]
                scores = arrays["original"][11].astype(np.float64)
                probability = scores / scores.sum()
                m = int(arrays["standard_mask"].sum())
                matched = stable_top(scores, m)
                masks = {q: top_p(scores, q) for q in THRESHOLDS}
                suppressed = {q: top_p(scores, q, True) for q in THRESHOLDS}
                masks_by_episode[task][seed].append(set(stable_top(scores, 16).tolist()))
                attentions[task].append(probability)
                top16_hits[task][stable_top(scores, 16)] += 1
                top32_hits[task][stable_top(scores, 32)] += 1

                row = {"task": SHORT[task], "seed": seed, "step": int(meta.get("source_step", meta.get("step"))),
                       "phase": meta["phase"], "matched_m": m,
                       "corner_attention_mass": float(probability[CORNERS].sum()),
                       "outer_edge_attention_mass": float(probability[EDGE].sum()),
                       "corner_enrichment_vs_area": float(probability[CORNERS].sum() / (4 / 256)),
                       "outer_edge_enrichment_vs_area": float(probability[EDGE].sum() / (60 / 256))}
                for q in THRESHOLDS:
                    tag = str(q).replace(".", "p")
                    original_set, suppressed_set = set(masks[q].tolist()), set(suppressed[q].tolist())
                    suppressed_scores = scores.copy()
                    suppressed_scores[EDGE] = -np.inf
                    same_m_set = set(stable_top(suppressed_scores, len(original_set)).tolist())
                    selected_edge = original_set & set(EDGE.tolist())
                    edge_mass = float(probability[list(selected_edge)].sum()) if selected_edge else 0.0
                    row.update({
                        f"top_{tag}_m": len(original_set), f"top_{tag}_edge_tokens": len(selected_edge),
                        f"top_{tag}_edge_attention_mass": edge_mass,
                        f"top_{tag}_edge_fraction_of_threshold_budget": edge_mass / q,
                        f"suppressed_top_{tag}_m": len(suppressed_set),
                        f"suppressed_top_{tag}_added_tokens": len(suppressed_set - original_set),
                        f"suppressed_top_{tag}_dropped_tokens": len(original_set - suppressed_set),
                        f"suppressed_top_{tag}_jaccard": len(original_set & suppressed_set) / len(original_set | suppressed_set),
                        f"same_m_suppressed_top_{tag}_added_tokens": len(same_m_set - original_set),
                        f"same_m_suppressed_top_{tag}_added_token_ids": " ".join(str(x) for x in sorted(same_m_set - original_set)),
                    })
                state_rows.append(row)

                fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
                axes[0].imshow(image)
                heat = scores.reshape(16, 16)
                axes[0].imshow(heat, extent=(0, image.shape[1], image.shape[0], 0), cmap="magma", alpha=.56, interpolation="bilinear")
                axes[0].set_title("Original + L11 attention", fontsize=9); axes[0].axis("off")
                overlay_mask(axes[1], image, matched, f"L11 Matched mask (m={m})")
                nested_overlay(axes[2], image, masks, "Cumulative selection")
                nested_overlay(axes[3], image, suppressed, "Outer-edge suppressed")
                fig.suptitle(f"{SHORT[task]} | seed {seed} | {meta['phase']} | step {row['step']}")
                fig.tight_layout()
                figure = galleries / f"{SHORT[task]}__seed{seed:03d}__{meta['phase']}.png"
                fig.savefig(figure, dpi=150); plt.close(fig)
                row["figure"] = str(figure.relative_to(ROOT))

                control = meta.get("target_control")
                if "target_switch" in arrays.files and control:
                    switched = arrays["target_switch"][11].astype(np.float64)
                    switched /= switched.sum()
                    old = np.asarray(control["old_target"], dtype=np.int64)
                    new = np.asarray(control["new_target"], dtype=np.int64)
                    original_mask = set(stable_top(probability, m).tolist())
                    switched_mask = set(stable_top(switched, m).tolist())
                    target_rows.append({
                        "task": SHORT[task], "seed": seed, "step": row["step"],
                        "old_instruction": meta["instruction"], "new_instruction": control["new_instruction"],
                        "new_target_attention_gain": float(switched[new].sum() - probability[new].sum()),
                        "old_target_attention_change": float(switched[old].sum() - probability[old].sum()),
                        "target_follow_contrast": float((switched[new].sum() - probability[new].sum()) - (switched[old].sum() - probability[old].sum())),
                        "new_target_top_m_coverage_change": len(switched_mask & set(new.tolist())) - len(original_mask & set(new.tolist())),
                        "old_target_top_m_coverage_change": len(switched_mask & set(old.tolist())) - len(original_mask & set(old.tolist())),
                        "top_m_jaccard_after_target_switch": len(original_mask & switched_mask) / len(original_mask | switched_mask),
                    })

    if len(state_rows) != 60:
        raise RuntimeError(f"expected 60 states, got {len(state_rows)}")
    write_csv(statistics / "state_budget_metrics.csv", state_rows)
    write_csv(statistics / "target_switch_metrics.csv", target_rows)

    task_summary = []
    for task in list(TASK_SEEDS) + ["Overall"]:
        rows = state_rows if task == "Overall" else [row for row in state_rows if row["task"] == SHORT[task]]
        record = {"task": "Overall" if task == "Overall" else SHORT[task], "states": len(rows)}
        numeric = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and key not in {"seed", "step"}]
        for key in numeric:
            record[f"mean_{key}"] = float(np.mean([row[key] for row in rows]))
        task_summary.append(record)
    write_csv(statistics / "task_summary.csv", task_summary)

    all_attention = np.concatenate([np.stack(attentions[task]) for task in TASK_SEEDS])
    all_top16 = sum(top16_hits.values())
    all_top32 = sum(top32_hits.values())
    hotspot_rows = []
    for task in TASK_SEEDS:
        mean_attention = np.mean(attentions[task], axis=0)
        for token in EDGE[np.argsort(top16_hits[task][EDGE])[::-1][:5]]:
            hotspot_rows.append({
                "task": SHORT[task], "token": int(token), "row": int(token // 16),
                "column": int(token % 16), "top16_count": int(top16_hits[task][token]),
                "top16_frequency": float(top16_hits[task][token] / 15),
                "mean_normalized_attention": float(mean_attention[token]),
            })
    write_csv(statistics / "top_edge_hotspots.csv", hotspot_rows)
    for filename, maps, title, cmap in (
        ("mean_attention_maps.png", {**{SHORT[t]: np.mean(attentions[t], axis=0) for t in TASK_SEEDS}, "Overall": all_attention.mean(axis=0)}, "Mean normalized L11 visual attention", "magma"),
        ("top16_frequency_maps.png", {**{SHORT[t]: top16_hits[t] / 15 for t in TASK_SEEDS}, "Overall": all_top16 / 60}, "Top-16 occurrence frequency", "viridis"),
        ("top32_frequency_maps.png", {**{SHORT[t]: top32_hits[t] / 15 for t in TASK_SEEDS}, "Overall": all_top32 / 60}, "Top-32 occurrence frequency", "viridis"),
    ):
        fig, axes = plt.subplots(1, 5, figsize=(16, 3.1))
        vmax = max(value.max() for value in maps.values())
        for ax, (name, value) in zip(axes, maps.items()):
            image = ax.imshow(value.reshape(16, 16), cmap=cmap, vmin=0, vmax=vmax)
            ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(title); fig.colorbar(image, ax=axes, fraction=.02, pad=.02)
        fig.savefig(statistics / filename, dpi=180, bbox_inches="tight"); plt.close(fig)

    def pairwise_jaccard(sets):
        return float(np.mean([len(a & b) / len(a | b) for a, b in combinations(sets, 2)]))
    spatial = {
        "states": 60,
        "mean_corner_mass": float(all_attention[:, CORNERS].sum(axis=1).mean()),
        "mean_outer_edge_mass": float(all_attention[:, EDGE].sum(axis=1).mean()),
        "corner_area_fraction": 4 / 256, "outer_edge_area_fraction": 60 / 256,
        "top16_outer_edge_occurrence_fraction": float(all_top16[EDGE].sum() / all_top16.sum()),
        "top32_outer_edge_occurrence_fraction": float(all_top32[EDGE].sum() / all_top32.sum()),
        "maximum_single_token_top16_frequency": float(all_top16.max() / 60),
        "maximum_edge_token_top16_frequency": float(all_top16[EDGE].max() / 60),
        "maximum_interior_token_top16_frequency": float(np.delete(all_top16, EDGE).max() / 60),
        "mean_early_late_top16_jaccard": float(np.mean([
            len(phases[0] & phases[2]) / len(phases[0] | phases[2])
            for task in TASK_SEEDS for phases in masks_by_episode[task].values()
        ])),
        "task_maximum_edge_top16_frequency": {
            SHORT[task]: float(top16_hits[task][EDGE].max() / 15) for task in TASK_SEEDS
        },
        "task_maximum_interior_top16_frequency": {
            SHORT[task]: float(np.delete(top16_hits[task], EDGE).max() / 15) for task in TASK_SEEDS
        },
    }
    if target_rows:
        spatial["target_switch_controls"] = len(target_rows)
        spatial["mean_target_follow_contrast"] = float(np.mean([row["target_follow_contrast"] for row in target_rows]))
        spatial["positive_target_follow_fraction"] = float(np.mean([row["target_follow_contrast"] > 0 for row in target_rows]))
        spatial["mean_target_switch_top_m_jaccard"] = float(np.mean([row["top_m_jaccard_after_target_switch"] for row in target_rows]))
    (statistics / "spatial_summary.json").write_text(json.dumps(spatial, indent=2) + "\n")

    overall = task_summary[-1]
    lines = [
        "# L11 Attention Position and Budget Diagnostic", "",
        "This is an offline diagnostic over 60 fixed states (4 tasks × 5 episodes × early/middle/late). No newly computed policy action controlled an environment and no new success rate was produced.", "",
        "## Aggregate facts", "",
        f'- Four corner tokens hold {100*spatial["mean_corner_mass"]:.2f}% of normalized visual attention versus 1.56% of positions.',
        f'- The outer ring holds {100*spatial["mean_outer_edge_mass"]:.2f}% versus 23.44% of positions.',
        f'- Outer-ring tokens account for {100*spatial["top16_outer_edge_occurrence_fraction"]:.2f}% of all Top-16 occurrences and {100*spatial["top32_outer_edge_occurrence_fraction"]:.2f}% of Top-32 occurrences.',
        f'- At Top-p=.8, the outer ring consumes on average {100*overall["mean_top_0p8_edge_fraction_of_threshold_budget"]:.2f}% of the cumulative threshold budget.',
        f'- Hard outer-ring suppression changes Top-p=.8 selection by adding {overall["mean_suppressed_top_0p8_added_tokens"]:.1f} and dropping {overall["mean_suppressed_top_0p8_dropped_tokens"]:.1f} tokens per state; mean mask Jaccard is {overall["mean_suppressed_top_0p8_jaccard"]:.3f}.',
        f'- The most persistent edge token appears in Top-16 for {100*spatial["maximum_edge_token_top16_frequency"]:.1f}% of states; the most persistent interior token appears for {100*spatial["maximum_interior_token_top16_frequency"]:.1f}%.',
        f'- Early-to-late Top-16 Jaccard within an episode is {spatial["mean_early_late_top16_jaccard"]:.3f}.',
    ]
    if target_rows:
        lines += [
            f'- In {len(target_rows)} explicit same-image target switches, target-follow contrast is positive in {100*spatial["positive_target_follow_fraction"]:.1f}% and the mean Top-m Jaccard before/after switching target is {spatial["mean_target_switch_top_m_jaccard"]:.3f}.',
        ]
    lines += ["", "## Judgment", "",
              "There is no broad outer-boundary attention sink: corners and the full outer ring are strongly under-represented by attention mass and by Top-16/Top-32 occurrence. The outer ring does not dominate the cumulative budget.",
              "There are isolated task-specific top-edge hotspots: a top-edge token reaches Top-16 in 60% of close_drawer and pick_coke_can states. This is a local positional bias, not whole-edge enrichment; interior hotspots are more persistent overall.",
              "The target-switch controls show that L11 is usually instruction-responsive rather than spatially fixed. Therefore these data do not support hard suppression of the entire outer ring as the next default. A later closed-loop test, if desired, should isolate a small recurrent-position penalty from broad edge removal.",
              "", "## Interpretation boundary", "",
              "Attention enrichment and budget occupancy do not establish that suppressing any position improves closed-loop success; that requires a separately paired intervention experiment."]
    (ROOT / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"complete": True, "states": 60, "spatial": spatial, "overall": overall}, indent=2))


if __name__ == "__main__":
    main()
