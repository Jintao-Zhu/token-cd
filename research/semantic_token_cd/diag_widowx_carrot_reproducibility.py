"""Diagnostic: which widowx_carrot_on_plate seeds are non-reproducible across
process launches (canonical snapshot hash mismatch vs the frozen reference).

Reads the frozen ATTN_SEMANTIC_K32_L8_8TASK_50_V1 summaries and, for each seed,
re-captures the snapshot in THIS fresh process, then reports which of the three
hashes (canonical / state / rgb) differ from the reference. No episodes run.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
for p in (REPO_ROOT / "task1/shim_site", REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from research.semantic_token_cd.distractor_rollout import (  # noqa: E402
    capture_snapshot,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.spatial_grid_rollout import make_environment  # noqa: E402


TASK = "widowx_carrot_on_plate"
REFERENCE_ROOT = Path(
    "artifacts/attn_semantic_k32_l8_8task_50_v1/episodes/"
    "widowx_carrot_on_plate/attn_semantic"
)


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    env, _ = make_environment(TASK)
    bad = []
    for seed in range(50):
        ref = json.loads((REFERENCE_ROOT / f"episode_{seed:03d}_summary.json").read_text())
        snapshot = capture_snapshot(env, seed)
        canonical = snapshot_sha(snapshot)
        _, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
        fields = {
            "canonical": (canonical == ref["canonical_snapshot_sha256"], canonical),
            "state": (state_sha == ref["initial_state_sha256"], state_sha),
            "rgb": (rgb_sha == ref["initial_rgb_sha256"], rgb_sha),
        }
        fail = [k for k, (ok, _) in fields.items() if not ok]
        if fail:
            bad.append((seed, fail))
        print(json.dumps({"seed": seed, "fail": fail}, sort_keys=True), flush=True)
    print(json.dumps({"n_bad": len(bad), "bad": [s for s, _ in bad]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
