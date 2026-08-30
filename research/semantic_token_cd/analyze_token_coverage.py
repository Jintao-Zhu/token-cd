"""Project the target drawer's handle/front 3D positions into the overhead camera,
and compare against the KMeans-selected token cells. Answers Q1 precisely: do the
selected tokens cover the handle, the front panel, or background?

Camera (overhead_camera, fixed): intrinsic fx=fy=425, cx=320, cy=256 (OpenCV cv),
extrinsic_cv world->cam. Token grid: 16x16 over the 512x640 frame (each cell 32x40px).
"""
from __future__ import annotations

import argparse
import json
import glob
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
ARTIFACT = REPO_ROOT / "artifacts/attn_semantic_merge_k8_v1"
MERGE_ARM = "semantic_merge_k8_eta100"

# overhead_camera (fixed): world -> cam (OpenCV), then u=fx*x/z+cx, v=fy*y/z+cy
EXT = np.array([[0.0029, 1.0, -0.0, -0.0112],
                [0.7069, -0.002, -0.7073, 0.4857],
                [-0.7073, 0.002, -0.7069, 1.3422],
                [0.0, 0.0, 0.0, 1.0]])
FX, FY, CX, CY = 425.0, 425.0, 320.0, 256.0


def project(p_w):
    p = np.array([p_w[0], p_w[1], p_w[2], 1.0])
    c = EXT @ p
    u = FX * c[0] / c[2] + CX
    v = FY * c[1] / c[2] + CY
    return u, v


def tok_cell(u, v):
    return int(v // 32), int(u // 40)  # (row, col)


def selected_cells(seed):
    p = ARTIFACT / "episodes" / "google_robot_close_drawer" / MERGE_ARM / f"episode_{seed:03d}_summary.json"
    t0 = json.loads(p.read_text())["selector_trace"][0]
    ids = t0.get("selected_token_ids", [])
    return {(i // 16, i % 16) for i in ids}, t0.get("selected_entities", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ARTIFACT / "replay_close_drawer")
    a = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(str(a.out / "seed_*.json"))):
        seed = int(Path(f).stem.split("_")[1])
        d = json.load(open(f))
        van = d["arms"].get("vanilla", {})
        hp = van.get("handle_p")
        fp = van.get("front_p")
        if hp is None or fp is None:
            continue
        sel, ents = selected_cells(seed)
        u_h, v_h = project(hp)
        u_f, v_f = project(fp)
        r_h, c_h = tok_cell(u_h, v_h)
        r_f, c_f = tok_cell(u_f, v_f)
        handle_in_sel = (r_h, c_h) in sel
        front_in_sel = (r_f, c_f) in sel
        # distance (in cells) from handle cell to nearest selected cell
        d_h = min(abs(r - r_h) + abs(c - c_h) for r, c in sel) if sel else None
        rows.append({
            "seed": seed, "entities": ents,
            "handle_uv": [round(u_h, 1), round(v_h, 1)], "front_uv": [round(u_f, 1), round(v_f, 1)],
            "handle_cell": [r_h, c_h], "front_cell": [r_f, c_f],
            "handle_in_sel": handle_in_sel, "front_in_sel": front_in_sel,
            "handle_cell_dist_to_sel": d_h,
            "sel_rows": sorted({r for r, c in sel}), "sel_cols": sorted({c for r, c in sel}),
        })
    rows.sort(key=lambda r: r["seed"])
    for r in rows:
        print(json.dumps(r))
    (a.out / "token_coverage.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {a.out / 'token_coverage.json'} ({len(rows)} seeds)")


if __name__ == "__main__":
    main()
