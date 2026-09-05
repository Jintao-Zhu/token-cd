"""Render image-space projections of IC-SHR token masks for paired cases.

IC-SHR modifies visual tokens rather than RGB pixels.  These figures project the
16x16 token locations back onto the initial camera image for interpretation.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from research.semantic_token_cd.distractor_rollout import (
    get_image_from_maniskill2_obs_dict,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.spatial_grid_rollout import make_environment


DEFAULT_CASES = {
    "google_robot_close_drawer": {"rescue": 16, "harm": 17},
    "google_robot_open_drawer": {"rescue": 16, "harm": 0},
    "google_robot_pick_coke_can": {"rescue": 0, "harm": 5},
    "google_robot_move_near": {"rescue": 28, "harm": 2},
}


def token_mask(token_ids: list[int]) -> np.ndarray:
    mask = np.zeros((16, 16), dtype=bool)
    if token_ids:
        ids = np.asarray(token_ids, dtype=int)
        mask[ids // 16, ids % 16] = True
    return mask


def overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha=.48):
    out = rgb.astype(np.float32).copy()
    h, w = out.shape[:2]
    for row, col in np.argwhere(mask):
        y0, y1 = round(row * h / 16), round((row + 1) * h / 16)
        x0, x1 = round(col * w / 16), round((col + 1) * w / 16)
        out[y0:y1, x0:x1] = (1 - alpha) * out[y0:y1, x0:x1] + alpha * np.asarray(color)
        out[y0:y0 + 2, x0:x1] = color
        out[max(y1 - 2, y0):y1, x0:x1] = color
        out[y0:y1, x0:x0 + 2] = color
        out[y0:y1, max(x1 - 2, x0):x1] = color
    return np.clip(out, 0, 255).astype(np.uint8)


def cutout(rgb: np.ndarray, mask: np.ndarray):
    out = np.full_like(rgb, 245)
    h, w = out.shape[:2]
    for row, col in np.argwhere(mask):
        y0, y1 = round(row * h / 16), round((row + 1) * h / 16)
        x0, x1 = round(col * w / 16), round((col + 1) * w / 16)
        out[y0:y1, x0:x1] = rgb[y0:y1, x0:x1]
    return out


def render_case(root: Path, snapshots: Path, out: Path, task: str, category: str, seed: int):
    summary_path = root / "episode_summary" / task / "ic_shr" / f"episode_{seed:03d}_summary.json"
    mask_path = root / "masks" / task / f"episode_{seed:03d}_components.json"
    summary = json.loads(summary_path.read_text())
    trace = json.loads(mask_path.read_text())["steps"][0]
    before = token_mask(trace["original_group_tokens"])
    after_ids = [x for component in trace["selected_components"] for x in component]
    after = token_mask(after_ids)
    removed = before & ~after

    with (snapshots / "snapshots" / task / f"seed_{seed:03d}.pkl").open("rb") as handle:
        snapshot = pickle.load(handle)
    if snapshot_sha(snapshot) != summary["canonical_snapshot_sha256"]:
        raise RuntimeError(f"snapshot mismatch: {task} seed {seed}")
    env, _ = make_environment(task)
    try:
        obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
        if state_sha != summary["initial_state_sha256"] or rgb_sha != summary["initial_rgb_sha256"]:
            raise RuntimeError(f"initial state/RGB mismatch: {task} seed {seed}")
        rgb = get_image_from_maniskill2_obs_dict(env, obs)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    finally:
        env.close()

    panels = [
        (rgb, "Original image"),
        (overlay(rgb, before, (235, 55, 45)), f"Original SHR group G ({before.sum()} tokens)"),
        (overlay(rgb, after, (25, 180, 70)), f"IC-SHR retained G' ({after.sum()} tokens)"),
        (overlay(rgb, removed, (35, 110, 235)), f"Filtered components G-G' ({removed.sum()} tokens)"),
        (cutout(rgb, after), "Image patches corresponding to IC mask"),
        (cutout(rgb, removed), "Image patches removed by component selection"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (image, title) in zip(axes.ravel(), panels):
        ax.imshow(image); ax.set_title(title, fontsize=11); ax.axis("off")
    fig.suptitle(
        f"{task.replace('google_robot_', '')} | seed {seed:03d} | IC-SHR vs SHR: {category}\n"
        f"instruction: {summary['instruction']} | SHR={summary['reference_success']} | IC-SHR={summary['success']}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, .94))
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{task.replace('google_robot_', '')}_seed_{seed:03d}_{category}.png"
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"task": task, "seed": seed, "category": category, "output": str(target)}))
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, default=Path("artifacts/instruction_component_shr_v1"))
    ap.add_argument("--snapshots", type=Path, default=Path("artifacts/vanilla_recon_shr_canonical_0_299_v2"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/instruction_component_shr_v1/visualizations"))
    args = ap.parse_args()
    for task, categories in DEFAULT_CASES.items():
        for category, seed in categories.items():
            render_case(args.artifact, args.snapshots, args.out, task, category, seed)


if __name__ == "__main__":
    main()
