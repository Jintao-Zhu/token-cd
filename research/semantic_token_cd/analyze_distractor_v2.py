#!/usr/bin/env python3
"""Preregistered analysis for SIMPLER Distractor Semantic Entity-CD V2."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


EXPERIMENT = "SIMPLER_DISTRACTOR_SEMANTIC_ENTITY_CD_PHASE0_V2"
ARMS = ("vanilla", "random_cd", "semantic_cd")


def load_episodes(artifact: Path):
    episodes = {}
    for path in sorted((artifact / "episodes").glob("*/*/episode_*_summary.json")):
        row = json.loads(path.read_text())
        episodes[(row["task"], int(row["seed"]), row["arm"])] = row
    return episodes


def success_rate(episodes, arm, task=None):
    values = [
        int(row["success"])
        for (row_task, _seed, row_arm), row in episodes.items()
        if row_arm == arm and (task is None or row_task == task)
    ]
    return (float(np.mean(values)), len(values)) if values else (float("nan"), 0)


def paired_counts(episodes, reference, candidate):
    counts = {"rescue": 0, "harm": 0, "fail_fail": 0, "succ_succ": 0}
    keys = sorted({(task, seed) for task, seed, arm in episodes if arm == reference})
    for task, seed in keys:
        left = episodes.get((task, seed, reference))
        right = episodes.get((task, seed, candidate))
        if left is None or right is None:
            continue
        pair = (bool(left["success"]), bool(right["success"]))
        if pair == (False, True):
            counts["rescue"] += 1
        elif pair == (True, False):
            counts["harm"] += 1
        elif pair == (False, False):
            counts["fail_fail"] += 1
        else:
            counts["succ_succ"] += 1
    counts["n"] = sum(counts.values())
    counts["ratio"] = (
        counts["rescue"] / counts["harm"]
        if counts["harm"]
        else (float("inf") if counts["rescue"] else float("nan"))
    )
    return counts


def verify_integrity(artifact, episodes, tasks):
    expected = 3 * 50 * 3
    errors = []
    if len(episodes) != expected:
        errors.append(f"expected {expected} summaries, found {len(episodes)}")
    for task in tasks:
        for seed in range(50):
            rows = [episodes.get((task, seed, arm)) for arm in ARMS]
            if any(row is None for row in rows):
                errors.append(f"missing triple: {task} seed {seed}")
                continue
            if len({row["initial_state_sha256"] for row in rows}) != 1:
                errors.append(f"state pairing mismatch: {task} seed {seed}")
            if len({row["initial_rgb_sha256"] for row in rows}) != 1:
                errors.append(f"RGB pairing mismatch: {task} seed {seed}")
            random = rows[ARMS.index("random_cd")]["selector_trace"]
            for step, rnd in enumerate(random):
                # Arms may follow different trajectories after the first action.
                # Audit random against its same-observation semantic reference.
                if rnd["num_groups"] != len(rnd["selected_group_ids"]):
                    errors.append(f"group count mismatch: {task} seed {seed} step {step}")
                if rnd["num_tokens"] != sum(rnd["selected_group_sizes"]):
                    errors.append(f"group sizes mismatch: {task} seed {seed} step {step}")
                if rnd["num_tokens"] != len(rnd["reference_semantic_token_ids"]):
                    errors.append(f"semantic token reference mismatch: {task} seed {seed} step {step}")
                if rnd["num_tokens"] != len(rnd["selected_token_ids"]):
                    errors.append(f"random token count mismatch: {task} seed {seed} step {step}")
            for arm, row in zip(ARMS, rows):
                logits = artifact / "episodes" / task / arm / row["logits_file"]
                if not logits.exists():
                    errors.append(f"missing logits: {logits}")
    return {"PASS": not errors, "errors": errors[:100], "n_errors": len(errors)}


def draw_token_map(image_path: Path, token_ids, color, label):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    for token_id in token_ids:
        row, col = divmod(int(token_id), 16)
        box = (
            col * width / 16,
            row * height / 16,
            (col + 1) * width / 16,
            (row + 1) * height / 16,
        )
        draw.rectangle(box, fill=(*color, 95), outline=(*color, 220), width=1)
    draw.rectangle((0, 0, width, 22), fill=(0, 0, 0, 190))
    draw.text((5, 4), label, fill=(255, 255, 255, 255))
    return image


def make_visualizations(artifact, episodes, tasks):
    out = artifact / "visualizations"
    out.mkdir(exist_ok=True)
    keys = [(task, seed) for task in tasks for seed in range(50)]
    rng = np.random.default_rng(20260824)
    selected = sorted(keys[index] for index in rng.choice(len(keys), size=20, replace=False))
    manifest = []
    for task, seed in selected:
        image_path = artifact / "initial_states" / task / f"episode_{seed:03d}.png"
        original = Image.open(image_path).convert("RGB")
        semantic = episodes[(task, seed, "semantic_cd")]
        random = episodes[(task, seed, "random_cd")]
        sem_trace = semantic["selector_trace"][0]
        rnd_trace = random["selector_trace"][0]
        sem_map = draw_token_map(
            image_path,
            sem_trace["selected_token_ids"],
            (0, 190, 100),
            f"Semantic | success={semantic['success']}",
        )
        rnd_map = draw_token_map(
            image_path,
            rnd_trace["selected_token_ids"],
            (225, 70, 45),
            f"Random | success={random['success']}",
        )
        status = original.copy()
        status_draw = ImageDraw.Draw(status, "RGBA")
        status_draw.rectangle((0, 0, status.width, 62), fill=(0, 0, 0, 205))
        status_draw.multiline_text(
            (5, 4),
            f"van={episodes[(task, seed, 'vanilla')]['success']}  "
            f"sem={semantic['success']}  rnd={random['success']}\n"
            f"groups={sem_trace['selected_group_ids']} tokens={sem_trace['num_tokens']}",
            fill=(255, 255, 255, 255),
            spacing=4,
        )
        canvas = Image.new("RGB", (original.width * 4, original.height))
        for index, panel in enumerate((original, sem_map, rnd_map, status)):
            canvas.paste(panel, (index * original.width, 0))
        filename = f"{task}_episode_{seed:03d}.png"
        canvas.save(out / filename)
        manifest.append({"task": task, "seed": seed, "file": filename})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    episodes = load_episodes(artifact)
    tasks = sorted({task for task, _seed, _arm in episodes})
    if not tasks:
        raise RuntimeError("No rollout summaries found")

    integrity = verify_integrity(artifact, episodes, tasks)
    sr = {arm: success_rate(episodes, arm)[0] for arm in ARMS}
    task_sr = {
        task: {arm: success_rate(episodes, arm, task)[0] for arm in ARMS}
        for task in tasks
    }
    semantic_random_gap = sr["semantic_cd"] - sr["random_cd"]
    semantic_rh = paired_counts(episodes, "vanilla", "semantic_cd")
    random_rh = paired_counts(episodes, "vanilla", "random_cd")
    task_wins = sum(
        task_sr[task]["semantic_cd"] > task_sr[task]["random_cd"]
        for task in tasks
    )
    gates = {
        "A": {"PASS": semantic_random_gap > 0.03, "gap": semantic_random_gap, "threshold": 0.03},
        "B": {"PASS": semantic_rh["ratio"] > 1.5, **semantic_rh, "threshold": 1.5},
        "C": {"PASS": task_wins >= 2, "semantic_wins": task_wins, "of": len(tasks), "threshold": 2},
        "integrity": integrity,
    }
    status = (
        "PASS_SEMANTIC_INDEPENDENT_VALUE"
        if integrity["PASS"] and all(gates[key]["PASS"] for key in ("A", "B", "C"))
        else "STOP_SEMANTIC_NOT_DISTINGUISHED_FROM_RANDOM"
    )
    decision = {
        "experiment": EXPERIMENT,
        "status": status,
        "n_tasks": len(tasks),
        "n_episodes_per_arm": success_rate(episodes, "vanilla")[1],
        "SR": sr,
        "semantic_minus_random": semantic_random_gap,
        "semantic_rescue_harm": semantic_rh,
        "random_rescue_harm": random_rh,
        "task_SR": task_sr,
        "gates": gates,
    }
    (artifact / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    with (artifact / "task_breakdown.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "vanilla", "random_cd", "semantic_cd", "semantic_minus_random"])
        for task in tasks:
            row = task_sr[task]
            writer.writerow([task, row["vanilla"], row["random_cd"], row["semantic_cd"], row["semantic_cd"] - row["random_cd"]])
    with (artifact / "rescue_harm.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "rescue", "harm", "ratio", "fail_fail", "succ_succ", "n"])
        for arm, row in (("semantic_cd", semantic_rh), ("random_cd", random_rh)):
            writer.writerow([arm, row["rescue"], row["harm"], row["ratio"], row["fail_fail"], row["succ_succ"], row["n"]])

    make_visualizations(artifact, episodes, tasks)
    lines = [
        f"# {EXPERIMENT}",
        "",
        f"> Verdict: `{status}`",
        "",
        "## Core result",
        "",
        f"- Vanilla SR: {sr['vanilla']:.3f}",
        f"- Random-CD SR: {sr['random_cd']:.3f}",
        f"- Semantic-CD SR: {sr['semantic_cd']:.3f}",
        f"- Semantic - Random: {semantic_random_gap:+.3f}",
        f"- Semantic Rescue/Harm: {semantic_rh['rescue']}/{semantic_rh['harm']} = {semantic_rh['ratio']:.3f}",
        "",
        "## Preregistered gates",
        "",
        "| Gate | Result | Value |",
        "|---|---|---|",
        f"| A: Semantic-Random >3pp | {'PASS' if gates['A']['PASS'] else 'FAIL'} | {semantic_random_gap:+.3f} |",
        f"| B: Rescue/Harm >1.5 | {'PASS' if gates['B']['PASS'] else 'FAIL'} | {semantic_rh['ratio']:.3f} |",
        f"| C: Semantic wins >=2/3 tasks | {'PASS' if gates['C']['PASS'] else 'FAIL'} | {task_wins}/{len(tasks)} |",
        f"| Integrity | {'PASS' if integrity['PASS'] else 'FAIL'} | errors={integrity['n_errors']} |",
        "",
        "## Task breakdown",
        "",
        "| task | vanilla | random | semantic | semantic-random |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in tasks:
        row = task_sr[task]
        lines.append(f"| {task} | {row['vanilla']:.3f} | {row['random_cd']:.3f} | {row['semantic_cd']:.3f} | {row['semantic_cd'] - row['random_cd']:+.3f} |")
    (artifact / "REPORT_ZH.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": status, "SR": sr, "semantic_minus_random": semantic_random_gap, "gates": gates}, ensure_ascii=False))


if __name__ == "__main__":
    main()
