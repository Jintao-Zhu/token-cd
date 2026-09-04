"""SC-SHR vs SHR RGB overlay figures (diagnostic, no new rollout)."""

from __future__ import annotations

import csv
import json
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


GRID = 16
FOLDER = {
    "google_robot_close_drawer": "drawer",
    "google_robot_open_drawer": "drawer",
    "google_robot_pick_coke_can": "coke",
    "google_robot_move_near": "move_near",
}
TASK_NAMES = {
    "google_robot_close_drawer": "close_drawer",
    "google_robot_open_drawer": "open_drawer",
    "google_robot_pick_coke_can": "pick_coke_can",
    "google_robot_move_near": "move_near",
}


def connected(token_ids):
    nodes = set(int(i) for i in token_ids)
    result = []
    while nodes:
        start = min(nodes)
        stack = [start]
        nodes.discard(start)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            r, c = divmod(u, GRID)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                nb = rr * GRID + cc
                if 0 <= rr < GRID and 0 <= cc < GRID and nb in nodes:
                    nodes.discard(nb)
                    stack.append(nb)
        result.append(comp)
    return result


def filter_keep(token_ids, n_keep):
    comps = sorted(connected(token_ids), key=len, reverse=True)
    kept = comps[: max(0, n_keep)]
    return sorted(int(i) for c in kept for i in c), [len(c) for c in comps]


def overlay(img, mask_ids, color, alpha=0.45):
    m = np.zeros(GRID * GRID, dtype=bool)
    m[np.asarray(mask_ids, dtype=np.int64)] = True
    up = np.kron(m.reshape(GRID, GRID),
                 np.ones((img.shape[0] // GRID, img.shape[1] // GRID)))
    layer = np.zeros_like(img, dtype=np.float32)
    for i, ch in enumerate((0, 1, 2)):
        layer[..., ch] = color[i]
    blend = img.astype(np.float32) / 255.0
    out = blend * (1 - alpha) + layer * alpha
    out = out * (1 - up[..., None]) + blend * up[..., None]
    return np.clip(out, 0, 1)


def main() -> None:
    base = Path("/data/docker/dev_zjt/data/code/artifacts/sc_shr_rgb_overlay_analysis")
    v1 = Path("/data/docker/dev_zjt/data/code/artifacts/spatial_harmonic_recon_v1/episodes")
    sc = Path("/data/docker/dev_zjt/data/code/artifacts/sc_shr_local_v1/episodes")
    cases = json.loads((base / "case_list.json").read_text())
    summary_rows = []

    for c in cases:
        task, seed = c["task"], c["seed"]
        rgb_path = base / "images" / task / f"seed_{seed:03d}.png"
        if not rgb_path.exists():
            continue
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        sc_sum = json.loads((sc / task / "sc_shr_harmonic"
                             / f"episode_{seed:03d}_summary.json").read_text())
        shr_sum = json.loads((v1 / task / "spatial_harmonic_recon"
                              / f"episode_{seed:03d}_summary.json").read_text())
        s_t0 = (shr_sum.get("selector_trace") or [{}])[0]
        c_t0 = (sc_sum.get("selector_trace") or [{}])[0]
        shr_ids = list(s_t0.get("selected_token_ids") or [])
        n_entities = max(1, len(s_t0.get("selected_entities") or []))
        n_keep = 1 if n_entities <= 1 else 2
        sc_ids, comp_sizes = filter_keep(shr_ids, n_keep)
        removed = sorted(set(shr_ids) - set(sc_ids))

        fig, axes = plt.subplots(1, 4, figsize=(22, 6))
        axes[0].imshow(rgb)
        axes[0].axis("off")
        axes[0].set_title("(1) original RGB", fontsize=11)
        axes[1].imshow(overlay(rgb, shr_ids, (1.0, 0.0, 0.0)))
        axes[1].axis("off")
        axes[1].set_title("(2) SHR mask (red)", fontsize=11)
        axes[2].imshow(overlay(rgb, sc_ids, (0.0, 1.0, 0.0)))
        axes[2].axis("off")
        axes[2].set_title("(3) SC-SHR mask (green)", fontsize=11)
        axes[3].imshow(overlay(rgb, removed, (0.1, 0.3, 1.0)))
        axes[3].axis("off")
        axes[3].set_title("(4) removed by SC filter (blue)", fontsize=11)

        instruction = sc_sum.get("instruction", "")
        fig.suptitle(
            f"task: {TASK_NAMES[task]}  seed:{seed}\n"
            f"{'SHR success' if shr_sum['success'] else 'SHR fail'} -> "
            f"{'SC-SHR success' if sc_sum['success'] else 'SC-SHR fail'}\n"
            f"SHR tokens:{len(shr_ids)}  SC tokens:{len(sc_ids)}  "
            f"removed:{len(removed)}\n"
            f"instruction: {instruction}",
            fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.84))
        folder = base / "images" / FOLDER[task]
        folder.mkdir(parents=True, exist_ok=True)
        out_png = folder / f"{c['category']}_seed{seed:03d}.png"
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
        summary_rows.append({
            "task": TASK_NAMES[task], "seed": seed, "category": c["category"],
            "SHR_tokens": len(shr_ids), "SC_tokens": len(sc_ids),
            "removed_tokens": len(removed),
            "removed_ratio": round(len(removed) / max(1, len(shr_ids)), 3),
            "SHR_success": shr_sum["success"], "SC_success": sc_sum["success"],
        })
        print(json.dumps(summary_rows[-1]))

    with (base / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"figures+summary: {len(summary_rows)} cases -> {base}")


if __name__ == "__main__":
    main()
