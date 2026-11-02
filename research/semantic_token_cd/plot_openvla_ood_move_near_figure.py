#!/usr/bin/env python3
"""Plot paired OpenVLA OOD results with synchronized Move Near examples."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/openvla_ood_robustness_v3"
PREFLIGHT = ROOT / "artifacts/openvla_ood_robustness_v3_preflight"
OUT = ART / "figures"
TASK = "google_robot_move_near"
TASKS = ["google_robot_open_drawer", "google_robot_close_drawer",
         "google_robot_pick_coke_can", "google_robot_move_near"]
SEEDS = range(200, 300)
CONDITIONS = ["clean", "distractor", "texture", "illumination", "viewpoint", "composite"]
LABELS = ["Original", "Distractor", "Table texture", "Illumination", "Viewpoint", "Composite"]
ARMS = ["vanilla", "pcd", "ours"]
METHODS = ["Vanilla", "PCD", "L11-Matched"]
# Matched to the user's pastel visualization palette.
COLORS = ["#F4A7B9", "#E1829D", "#8CB3D9"]


def read_episode(condition: str, arm: str, seed: int) -> dict:
    path = ART / "episodes" / TASK / condition / arm / f"episode_{seed}_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def audit_data() -> dict[str, dict[str, int]]:
    counts = {}
    for condition in CONDITIONS:
        arm_successes = {arm: 0 for arm in ARMS}
        for task in TASKS:
            paired = {arm: {} for arm in ARMS}
            for arm in ARMS:
                rows = []
                for seed in SEEDS:
                    path = ART / "episodes" / task / condition / arm / f"episode_{seed}_summary.json"
                    row = json.loads(path.read_text())
                    if (int(row["seed"]) != seed or row["condition"] != condition
                            or row["arm"] != arm or row["task"] != task):
                        raise RuntimeError(f"Episode identity mismatch: {task}/{condition}/{arm}/{seed}")
                    rows.append(row)
                arm_successes[arm] += sum(bool(row["result"]["success"]) for row in rows)
                paired[arm] = {int(row["seed"]): row for row in rows}
            # Confirm all arms are paired on the same initial state and shifted RGB.
            for seed in SEEDS:
                rows = [paired[arm][seed] for arm in ARMS]
                for key in ("initial_state_sha256", "initial_rgb_sha256", "canonical_snapshot_sha256"):
                    if len({row[key] for row in rows}) != 1:
                        raise RuntimeError(f"Non-paired {key}: {task}/{condition}, seed {seed}")
        counts[condition] = arm_successes
    return counts


def load_image(condition: str) -> np.ndarray:
    path = PREFLIGHT / "audit_images" / TASK / f"seed_200_{condition}.png"
    if not path.is_file():
        raise FileNotFoundError(path)
    rgb = np.asarray(Image.open(path).convert("RGB"))
    digest = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
    formal = read_episode(condition, "vanilla", 200)
    if digest != formal["initial_rgb_sha256"]:
        raise RuntimeError(f"Displayed image is not the formal seed-200 input for {condition}")
    return rgb


def make_figure(counts: dict[str, dict[str, int]]) -> plt.Figure:
    fig = plt.figure(figsize=(16.2, 8.4), dpi=180, facecolor="white")
    gs = fig.add_gridspec(2, 6, height_ratios=(3.2, 1.45),
                          left=0.075, right=0.985, top=0.90, bottom=0.10,
                          hspace=0.10, wspace=0.10)
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(CONDITIONS))
    width = 0.23
    for j, (arm, label, color) in enumerate(zip(ARMS, METHODS, COLORS)):
        values = np.asarray([100 * counts[c][arm] / 400 for c in CONDITIONS])
        lows, highs = [], []
        for c in CONDITIONS:
            lo, hi = wilson(counts[c][arm], 400)
            lows.append(100 * lo); highs.append(100 * hi)
        pos = x + (j - 1) * width
        ax.bar(pos, values, width, color=color, label=label, zorder=3,
               edgecolor="white", linewidth=0.8,
               yerr=[values - np.asarray(lows), np.asarray(highs) - values],
               error_kw=dict(ecolor="#333333", lw=0.85, capsize=2.4, capthick=0.8))
        # Keep value labels above the full Wilson interval, not on top of its
        # whisker/cap. This matters most when the three bars in a group are close.
        for k, (xx, yy, condition) in enumerate(zip(pos, values, CONDITIONS)):
            ax.text(xx, highs[k] + 1.1, f"{yy:.1f}%", ha="center", va="bottom",
                    fontsize=9.5, color="#111111")
    ax.set_xlim(-0.55, len(CONDITIONS) - 0.45)
    ax.set_ylim(0, 72)
    ax.set_ylabel("Success rate (%)", fontsize=13, color="#111111")
    ax.set_yticks(np.arange(0, 81, 20))
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=11, colors="#111111")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7, zorder=0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#222222"); ax.spines["bottom"].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.10),
              fontsize=12, handlelength=1.4, columnspacing=2.0)

    for i, (condition, label) in enumerate(zip(CONDITIONS, LABELS)):
        image_ax = fig.add_subplot(gs[1, i])
        image_ax.imshow(load_image(condition), interpolation="lanczos")
        image_ax.set_title(label, fontsize=10.5, color="#111111", pad=8)
        image_ax.set_xticks([]); image_ax.set_yticks([])
        for spine in image_ax.spines.values():
            spine.set_color("#C6C6C6"); spine.set_linewidth(0.65)

    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "Nimbus Roman", "font.serif": ["Nimbus Roman", "Liberation Serif"],
        "font.size": 10, "axes.labelcolor": "#111111", "text.color": "#111111",
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    counts = audit_data()
    fig = make_figure(counts)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"ood_move_near_robustness.{ext}", dpi=360, bbox_inches="tight")
    # A download-friendly PNG stays below common ~2 MB transfer limits while
    # preserving enough resolution for paper drafts and slides.
    full_png = OUT / "ood_move_near_robustness.png"
    download_png = OUT / "ood_move_near_robustness_download.png"
    with Image.open(full_png) as source:
        width = 2885
        height = round(source.height * width / source.width)
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        source.convert("RGB").resize((width, height), resampling).save(
            download_png, format="PNG", optimize=True, compress_level=9
        )
    with Image.open(download_png) as check:
        check.verify()
    print(json.dumps({"counts": counts, "outputs": [str(OUT / f"ood_move_near_robustness.{e}")
                                                        for e in ("png", "pdf", "svg")]}, indent=2))
    plt.close(fig)


if __name__ == "__main__":
    main()
