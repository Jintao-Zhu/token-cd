"""Is the SIMPLER RT renderer bit-deterministic for an identical reset?

The 4500-episode run shows identical initial physics state across all five arms (73/73 groups)
but differing initial RGB hashes in 28/73 groups. That points at renderer non-determinism
rather than a reset problem. This renders the same seed twice in one process and compares the
frame hash, which separates "the renderer is stochastic" from "the reset differs".

Also reports the actual pixel difference so the magnitude is known, not just the hash.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

ROOT = Path("/home/leju-suzhou/zjt_ws")
sys.path.insert(0, str(ROOT / "token-cd"))
sys.path.insert(0, str(ROOT / "openpi/packages/openpi-client/src"))
sys.path.insert(0, str(ROOT / "openpi/src"))

import numpy as np  # noqa: E402


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main() -> None:
    task = os.environ.get("DET_TASK", "widowx_spoon_on_towel")
    seed = int(os.environ.get("DET_SEED", "3"))
    depth = int(os.environ.get("DET_RT_DEPTH", "8"))
    from research.semantic_token_cd.pi0_fast_simpler_l6_matched_rollout import _environment

    env = _environment(task, depth)
    cam = "overhead_camera" if "google_robot" in env.unwrapped.robot_uid else "3rd_view_camera"

    frames = []
    for i in range(3):
        obs, info = env.reset(seed=seed)
        f = np.asarray(obs["image"][cam]["rgb"])
        frames.append(f)
        print(f"[det] reset {i}: shape={f.shape} dtype={f.dtype} sha={digest(f)[:16]}", flush=True)

    for i in range(1, len(frames)):
        d = np.abs(frames[0].astype(int) - frames[i].astype(int))
        print(f"[det] reset0 vs reset{i}: max|Δ|={d.max()} mean|Δ|={d.mean():.4f} "
              f"changed_px={(d > 0).sum()}/{d.size} ({100*(d>0).mean():.2f}%)", flush=True)

    same = len({digest(f) for f in frames}) == 1
    print(f"[det] VERDICT: renderer is {'DETERMINISTIC' if same else 'NON-DETERMINISTIC'} "
          f"for identical reset", flush=True)
    env.close()


if __name__ == "__main__":
    main()
