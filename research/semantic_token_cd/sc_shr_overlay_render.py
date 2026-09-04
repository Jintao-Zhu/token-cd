"""Render initial RGB frames for the SC-SHR overlay analysis (no policy).

Reuses local rollout snapshots: for each (task, seed) the environment restores
the same canonical snapshot used by the SC-SHR rollout and the RGB is saved only
when its SHA256 matches the stored summary. No model inference is run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from research.semantic_token_cd.distractor_rollout import (
    capture_snapshot,
    get_image_from_maniskill2_obs_dict,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.sc_shr_local_rollout import parse_seeds
from research.semantic_token_cd.spatial_grid_rollout import make_environment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-list", default=(
        "artifacts/sc_shr_rgb_overlay_analysis/case_list.json"))
    ap.add_argument("--out", default="artifacts/sc_shr_rgb_overlay_analysis/images")
    ap.add_argument("--sc-artifact", default="artifacts/sc_shr_local_v1/episodes")
    args = ap.parse_args()

    cases = json.loads(Path(args.case_list).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    by_task: dict[str, list[dict]] = {}
    for c in cases:
        by_task.setdefault(c["task"], []).append(c)

    skipped = []
    for task, task_cases in by_task.items():
        env, _ = make_environment(task)
        for c in sorted(task_cases, key=lambda x: x["seed"]):
            seed = c["seed"]
            sc_summary = json.loads(
                (Path(args.sc_artifact) / task / "sc_shr_harmonic"
                 / f"episode_{seed:03d}_summary.json").read_text())
            snapshot = capture_snapshot(env, seed)
            if snapshot_sha(snapshot) != sc_summary["canonical_snapshot_sha256"]:
                skipped.append((task, seed, "canonical"))
                continue
            obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
            if state_sha != sc_summary["initial_state_sha256"] or \
               rgb_sha != sc_summary["initial_rgb_sha256"]:
                skipped.append((task, seed, "rgb"))
                continue
            rgb = get_image_from_maniskill2_obs_dict(env, obs)
            arr = rgb if rgb.dtype == "uint8" else (rgb * 255).astype("uint8")
            folder = out / task
            folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr).save(folder / f"seed_{seed:03d}.png")
            print(json.dumps({"task": task, "seed": seed, "rgb_ok": True}))
        env.close()

    (Path(args.out).parent / "render_skipped.json").write_text(
        json.dumps(skipped, indent=2))
    print(json.dumps({"rendered": 64 - len(skipped), "skipped": skipped},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
