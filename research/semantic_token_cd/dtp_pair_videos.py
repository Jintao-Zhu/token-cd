"""Build side-by-side calibration videos (control vs candidate) for each scene."""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from research.semantic_token_cd.dtp_closed_loop_protocol import ARTIFACT, TASKS, calibration_seeds

CANDIDATES = ("l11_k64_t05", "l7_k64_t05")


def frames_for(path: Path) -> list[np.ndarray]:
    reader = imageio.get_reader(path)
    return [np.asarray(frame, dtype=np.uint8) for frame in reader]


def main() -> None:
    root = ARTIFACT / "closed_loop"
    episodes_root = root / "episodes"
    paired_root = root / "paired_videos"
    pairs = 0
    for task in TASKS:
        for seed in calibration_seeds(task):
            video_dir = episodes_root / task / "videos"
            control_path = video_dir / "control" / f"episode_{seed:03d}.mp4"
            for candidate in CANDIDATES:
                cand_path = video_dir / candidate / f"episode_{seed:03d}.mp4"
                if not control_path.exists() or not cand_path.exists():
                    print(f"missing video: {task} seed={seed} {candidate}")
                    continue
                control = frames_for(control_path)
                cand = frames_for(cand_path)
                n = min(len(control), len(cand))
                rows = [np.concatenate([control[i], cand[i]], axis=1) for i in range(n)]
                out_dir = paired_root / task
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"episode_{seed:03d}_{candidate}.mp4"
                imageio.mimwrite(out_path, rows, fps=10, codec="libx264", quality=7,
                                 macro_block_size=None)
                pairs += 1
                print(out_path)
    print(f"paired videos written: {pairs}")


if __name__ == "__main__":
    main()
