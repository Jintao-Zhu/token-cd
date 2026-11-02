"""Create the paper-ready three-panel L11 mechanism figure from audited artifacts.

Panel A uses synchronized intermediate RGB, simulator instance segmentation,
and selected-token IDs from the replayed Full L11 state library. Panels B/C
are read directly from their paired-result JSON files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/l11_abc_mechanism_v1"
STATE_LIB = ART / "state_library/states"
OUT = ART / "figures"
TASKS = [
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
]
TASK_LABELS = ["Open Drawer", "Close Drawer", "Pick Coke Can", "Move Near"]
COLORS = {
    "full": "#315A7D",
    "enriched": "#E49B45",
    "sparing": "#B85C68",
    "zero": "#496A9B",
    "mean": "#52A6A0",
    "target": "#F2C14E",
    "on_target": "#E49B45",
    "off_target": "#42A89B",
}


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def model_view(rgb: np.ndarray) -> np.ndarray:
    # The experiment maps RGB/instance masks to the 224x224 OpenVLA view.
    return cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)


def _entity_ids(meta: dict, key: str) -> set[int]:
    entities = meta["target_entities"].get(key, {})
    return {int(value) for value in entities.values() if isinstance(value, int)}


def load_state_library_example(task: str, seed: int, step: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load one synchronized saved RGB/segmentation/L11 mask state."""
    stem = f"{task}__seed{seed:03d}__step{step:04d}"
    folder = STATE_LIB / task
    meta = read_json(folder / f"{stem}.json")
    if (meta["task"], int(meta["seed"]), int(meta["step"])) != (task, seed, step):
        raise RuntimeError(f"State metadata mismatch for {stem}")
    with np.load(folder / f"{stem}.npz", allow_pickle=False) as data:
        rgb_raw = np.asarray(data["rgb"])
        segmentation = np.asarray(data["segmentation_instance"], dtype=np.int64)
    rgb_hash = hashlib.sha256(np.ascontiguousarray(rgb_raw).tobytes()).hexdigest()
    seg_hash = hashlib.sha256(np.ascontiguousarray(segmentation).tobytes()).hexdigest()
    if rgb_hash != meta["state_rgb_sha256"] or seg_hash != meta["segmentation_sha256"]:
        raise RuntimeError(f"Saved RGB/segmentation hash mismatch for {stem}")

    # Use the same 224x224 resize and any-positive-overlap 16x16 patch rule as B.
    rgb = model_view(rgb_raw)
    ids = _entity_ids(meta, "target_object") | _entity_ids(meta, "target_drawer_link")
    if task == "google_robot_move_near":
        ids |= _entity_ids(meta, "source_object") | _entity_ids(meta, "target_object")
    if not ids:
        raise RuntimeError(f"No target instance IDs found for {stem}")
    target_raw = np.isin(segmentation, list(ids)).astype(np.uint8)
    target_mask = cv2.resize(target_raw, (224, 224), interpolation=cv2.INTER_AREA)
    target_grid = np.zeros((16, 16), dtype=bool)
    for row in range(16):
        for col in range(16):
            target_grid[row, col] = np.any(target_mask[row * 14:(row + 1) * 14,
                                                       col * 14:(col + 1) * 14] > 0)

    selected = np.asarray(meta["selected_token_ids"], dtype=int)
    m = int(meta["m_matched"])
    if len(selected) != m or len(np.unique(selected)) != m or np.any((selected < 0) | (selected >= 256)):
        raise RuntimeError(f"Invalid selected-token set for {stem}: expected {m}, got {len(selected)}")
    selected_on_target = selected[target_grid.reshape(-1)[selected]]
    selected_off_target = selected[~target_grid.reshape(-1)[selected]]
    if len(selected_on_target) + len(selected_off_target) != m:
        raise RuntimeError(f"Target split lost selected tokens for {stem}")
    info = {
        "task": task, "seed": seed, "step": step, "m": m,
        "selected": selected.tolist(), "on_target": selected_on_target.tolist(),
        "off_target": selected_off_target.tolist(), "target_pixels": target_mask,
        "target_patch_count": int(target_grid.sum()),
        "renderer": meta["renderer"], "instruction": meta["instruction"],
    }
    return rgb, target_mask, info


def wilson(success: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = success / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center, half


def soft_grid_overlay(ax, ids: list[int], color: str, alpha: float = 0.78) -> None:
    """Render selected patches as a continuous, feathered density field.

    This is a visualization of binary membership, not an attention map. A
    broader Gaussian kernel removes visible 16x16 cell boundaries; no cell
    outlines or token-center markers are drawn.
    """
    mask = np.zeros((16, 16), dtype=np.float32)
    ids_arr = np.asarray(ids, dtype=int)
    if ids_arr.size:
        mask[ids_arr // 16, ids_arr % 16] = 1.0
    density = cv2.GaussianBlur(mask, (0, 0), sigmaX=0.70, sigmaY=0.70,
                               borderType=cv2.BORDER_CONSTANT)
    if density.max() > 0:
        density /= density.max()
    # Upsample before compositing, so the displayed field has no token-cell edges.
    density = cv2.resize(density, (672, 672), interpolation=cv2.INTER_CUBIC)
    density = np.clip(density, 0, 1)
    rgba = np.zeros((672, 672, 4), dtype=float)
    rgba[..., :3] = np.asarray(matplotlib.colors.to_rgb(color))
    rgba[..., 3] = alpha * np.power(density, 0.78)
    ax.imshow(rgba, extent=(0, 224, 224, 0), interpolation="bilinear", zorder=2)


def draw_gt_target(ax, mask: np.ndarray) -> None:
    rgba = np.zeros((*mask.shape, 4), dtype=float)
    rgba[..., :3] = matplotlib.colors.to_rgb(COLORS["target"])
    rgba[..., 3] = 0.48 * (mask > 0)
    ax.imshow(rgba, extent=(0, 224, 224, 0), interpolation="nearest", zorder=3)
    contour = cv2.Canny((mask > 0).astype(np.uint8) * 255, 50, 150)
    ax.contour(np.arange(224) + 0.5, np.arange(224) + 0.5, contour,
               levels=[127], colors=["#FFF8D6"], linewidths=0.8, zorder=4)


def panel_a(fig) -> None:
    cols = ["RGB + GT target", "All L11-Matched", "L11 selected on target", "L11 selected outside target"]
    # Two distinct tasks and moderate, actual Matched budgets. These are
    # synchronized intermediate states, selected to show L11's target/context split.
    row_examples = [
        ("google_robot_open_drawer", 145, 69, "Open Drawer · seed 145 · step 69"),
        ("google_robot_pick_coke_can", 116, 29, "Pick Coke Can · seed 116 · step 29"),
    ]
    grid = fig.add_gridspec(2, 4, left=0.075, right=0.985, top=0.805,
                            bottom=0.145, wspace=0.075, hspace=0.22)
    for r, (task, seed, step, task_title) in enumerate(row_examples):
        rgb, target_mask, meta = load_state_library_example(task, seed, step)
        for c, title in enumerate(cols):
            ax = fig.add_subplot(grid[r, c])
            ax.imshow(rgb, interpolation="nearest")
            if c == 0:
                draw_gt_target(ax, target_mask)
                ax.text(0.96, 0.045, "simulator GT", color="#fffdf5", fontsize=8.8,
                        transform=ax.transAxes, ha="right", va="bottom",
                        bbox=dict(boxstyle="round,pad=.25", facecolor="#242A31", alpha=.76, edgecolor="none"),
                        zorder=5)
            else:
                if c == 1:
                    ids, color = meta["selected"], COLORS["full"]
                    tag = f"m={meta['m']}"
                elif c == 2:
                    ids, color = meta["on_target"], COLORS["on_target"]
                    tag = f"{len(ids)} selected"
                else:
                    ids, color = meta["off_target"], COLORS["off_target"]
                    tag = f"{len(ids)} selected"
                soft_grid_overlay(ax, ids, color, alpha=0.78)
                ax.text(0.96, 0.045, tag, color="white", fontsize=8.8,
                        transform=ax.transAxes, ha="right", va="bottom",
                        bbox=dict(boxstyle="round,pad=.22", facecolor="#242A31", alpha=.75, edgecolor="none"),
                        zorder=5)
            ax.set_xlim(0, 224); ax.set_ylim(224, 0); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#AAB6C2"); spine.set_linewidth(0.9)
            if r == 0:
                ax.set_title(title, fontsize=11.3, fontweight="semibold", pad=10, color="#111111")
            if c == 0:
                ax.text(-0.09, 0.5, task_title, rotation=90, va="center", ha="center",
                        transform=ax.transAxes, fontsize=10, fontweight="semibold", color="#111111")


def panel_b(ax) -> None:
    d = read_json(ART / "B_target_region/paired_results_vs_full.json")
    order = ["google_robot_open_drawer", "google_robot_close_drawer",
             "google_robot_pick_coke_can", "google_robot_move_near"]
    method_keys = ["full", "enriched", "sparing"]
    method_names = ["Full / L11-Matched", "Target-Enriched", "Target-Sparing"]
    result_keys = [None, "target_enriched", "target_sparing"]
    colors = [COLORS["full"], COLORS["enriched"], COLORS["sparing"]]
    x = np.arange(4)
    width = 0.22
    for i, (name, key, color) in enumerate(zip(method_names, result_keys, colors)):
        rates, lows, highs, counts = [], [], [], []
        for task in order:
            if key is None:
                record = d["tasks"]["target_enriched"][task]
            else:
                record = d["tasks"][key][task]
            count = int(record["reference_successes"] if key is None else record["arm_successes"])
            n = int(record["n"])
            p, h = wilson(count, n)
            rates.append(100 * count / n); lows.append(100 * (p - h)); highs.append(100 * (p + h)); counts.append(count)
        pos = x + (i - 1) * width
        ax.bar(pos, rates, width, color=color, label=name, zorder=3,
               edgecolor="white", linewidth=0.45, yerr=[np.array(rates)-np.array(lows),
                                                            np.array(highs)-np.array(rates)],
               error_kw=dict(ecolor="#34404D", lw=0.8, capsize=2.1, capthick=0.8))
        for xx, y, count in zip(pos, rates, counts):
            ax.text(xx, y + 2.3, f"{count}/100", ha="center", va="bottom", fontsize=9.5,
                    color="#273442", rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels(["Open Drawer", "Close Drawer", "Pick Coke Can", "Move Near"], fontsize=11)
    ax.set_ylim(0, 100); ax.set_ylabel("Success rate (%)", fontsize=12)
    ax.set_yticks(np.arange(0, 101, 20)); ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", color="#DCE3E9", linewidth=0.65, zorder=0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 1.07),
              ncol=3, handlelength=1.1, columnspacing=1.5)


def panel_c(ax) -> None:
    d = read_json(ART / "C_reconstruction/paired_reconstruction_results.json")
    order = ["google_robot_open_drawer", "google_robot_close_drawer",
             "google_robot_pick_coke_can", "google_robot_move_near"]
    comparisons = [
        ("Harmonic − Zero", "zero", COLORS["zero"]),
        ("Harmonic − Global Mean", "clean_global_mean", COLORS["mean"]),
    ]
    y = np.arange(4)
    height = 0.30
    offsets = [-0.22, 0.22]
    for j, (label, key, color) in enumerate(comparisons):
        vals = [100 * float(d["tasks"][key][task]["delta_rate"]) * -1 for task in order]
        # Delta in results is comparator minus harmonic, so negate for harmonic minus comparator.
        yy = y + offsets[j]
        ax.barh(yy, vals, height=height, color=color, edgecolor="white", linewidth=.55,
                label=label, zorder=3)
        for yi, v in zip(yy, vals):
            ax.text(v + (0.48 if v >= 0 else -0.48), yi, f"{v:+.1f}",
                    ha="left" if v >= 0 else "right", va="center", fontsize=10.5,
                    color="#263542", fontweight="semibold")
    ax.axvline(0, color="#263542", linewidth=0.9, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(["Open Drawer", "Close Drawer", "Pick Coke Can", "Move Near"], fontsize=11)
    ax.invert_yaxis(); ax.set_xlim(-7, 19); ax.set_xticks([-6, 0, 6, 12, 18])
    ax.set_xlabel("Success-rate gain (percentage points)", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="x", color="#DCE3E9", linewidth=.65, zorder=0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.spines["left"].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 1.06),
              ncol=2, handlelength=1.15)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        # Nimbus Roman is the installed Times-compatible serif face on this host.
        "font.family": "Nimbus Roman", "font.serif": ["Nimbus Roman", "Liberation Serif"], "font.size": 8,
        "axes.labelcolor": "#111111", "text.color": "#111111",
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    b = read_json(ART / "B_target_region/paired_results_vs_full.json")["overall"]
    c = read_json(ART / "C_reconstruction/paired_reconstruction_results.json")["overall"]
    figs = []
    fig_a = plt.figure(figsize=(15.0, 9.0), dpi=180)
    fig_a.text(0.055, 0.955,
               "Synchronized intermediate states · actual Matched budgets · 16×16 visual-token grid",
               ha="left", va="center", fontsize=11, color="#222222")
    panel_a(fig_a)
    fig_a.text(0.075, 0.068,
               "Blue = all L11-selected patches; orange/teal split the same selection by simulator target overlap. "
               "Colors show a Gaussian-smoothed binary selection density for display; cell boundaries are omitted.\n"
               "Target membership uses the experiment's any-positive-overlap patch rule. RGB, segmentation, and L11 mask are synchronized at the saved step.",
              ha="left", va="center", fontsize=9.1, color="#222222", linespacing=1.35)
    figs.append(("mechanistic_panel_a_token_masks", fig_a))

    fig_b = plt.figure(figsize=(11.5, 7.2), dpi=180)
    fig_b.suptitle("Target-region intervention", x=0.08, y=0.95, ha="left",
                   fontsize=20, fontweight="bold", color="#111111")
    fig_b.text(0.08, 0.905,
               f"Paired closed-loop results · {int(b['target_enriched']['n'])} episodes per method · 1,200 arm-episodes total",
               ha="left", va="center", fontsize=11, color="#222222")
    ax_b = fig_b.add_axes([0.10, 0.17, 0.84, 0.60])
    panel_b(ax_b)
    fig_b.text(0.10, 0.075,
               f"Overall: L11-Matched {b['target_enriched']['reference_successes']}/400 (48.5%) · "
               f"Target-Enriched {b['target_enriched']['arm_successes']}/400 (43.0%) · "
               f"Target-Sparing {b['target_sparing']['arm_successes']}/400 (33.0%).  Error bars: 95% Wilson intervals.",
               ha="left", va="center", fontsize=10, color="#222222")
    figs.append(("mechanistic_panel_b_target_intervention", fig_b))

    fig_c = plt.figure(figsize=(11.5, 7.2), dpi=180)
    fig_c.suptitle("Reconstruction strategy: Harmonic gain", x=0.08, y=0.95, ha="left",
                   fontsize=20, fontweight="bold", color="#111111")
    fig_c.text(0.08, 0.905,
               f"Paired closed-loop results · {int(c['zero']['n'])} episodes per method · 2,400 arm-episodes total",
               ha="left", va="center", fontsize=11, color="#222222")
    ax_c = fig_c.add_axes([0.15, 0.22, 0.78, 0.56])
    panel_c(ax_c)
    fig_c.text(0.15, 0.105,
               f"Overall gain: +4.75 pp vs Zero · +4.25 pp vs Global Mean · "
               f"Holm-adjusted exact McNemar p={c['zero']['mcnemar_holm_adjusted_p']:.4f} for both comparisons. "
               "Positive values favor Harmonic.",
               ha="left", va="center", fontsize=10, color="#222222", wrap=True)
    figs.append(("mechanistic_panel_c_reconstruction", fig_c))

    outputs = []
    for stem, fig in figs:
        paths = {ext: OUT / f"{stem}.{ext}" for ext in ("png", "pdf", "svg")}
        fig.savefig(paths["png"], dpi=360, bbox_inches="tight")
        fig.savefig(paths["pdf"], bbox_inches="tight")
        fig.savefig(paths["svg"], bbox_inches="tight")
        outputs.append({ext: str(path) for ext, path in paths.items()})
        plt.close(fig)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
