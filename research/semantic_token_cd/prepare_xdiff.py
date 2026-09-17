"""Freeze the semantic-difference experiment manifests before GPU work."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from research.semantic_token_cd.xdiff_protocol import (
    ARMS, ARTIFACT, ARM_CONFIG, EPSILON, SOURCE_ARTIFACT, TASKS, atomic_json,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source_manifest = SOURCE_ARTIFACT / "scene_manifest.json"
    scenes = json.loads(source_manifest.read_text())["scenes"]
    state_manifest = {"protocol": "PROMPT_ATTN_SEMANTIC_DIFFERENCE_V1", "states": {}}
    for task in TASKS:
        root = SOURCE_ARTIFACT / "runs" / "emitted_states" / task
        seeds = sorted(int(p.name.split("_")[1]) for p in root.glob("seed_*"))
        # Exactly 20 episodes/task, spread across the locked deduplicated list;
        # all three pre-fixed early/middle/late states stay together.
        positions = np.linspace(0, len(seeds) - 1, 20).round().astype(int)
        chosen = [seeds[i] for i in sorted(set(positions.tolist()))]
        if len(chosen) != 20:
            raise RuntimeError(f"could not choose 20 distinct episodes for {task}")
        entries = []
        for seed in chosen:
            for p in sorted((root / f"seed_{seed:03d}").glob("step_*.npz")):
                entries.append(str(p.relative_to(SOURCE_ARTIFACT)))
        if len(entries) != 60:
            raise RuntimeError(f"expected 60 states for {task}, got {len(entries)}")
        state_manifest["states"][task] = entries
    atomic_json(ARTIFACT / "STAGE2_STATE_MANIFEST.json", state_manifest)
    lock = {
        "protocol": "PROMPT_ATTN_SEMANTIC_DIFFERENCE_V1",
        "source_protocol": "PROMPT_ATTN_INSTR_SWAP_V1",
        "source_scene_manifest_sha256": sha(source_manifest),
        "state_manifest_sha256": sha(ARTIFACT / "STAGE2_STATE_MANIFEST.json"),
        "tasks": list(TASKS), "arms": list(ARMS), "new_arms": list(ARMS[2:]),
        "attention_layers": [11], "epsilon": EPSILON,
        "arm_config": ARM_CONFIG,
        "coverage": "own-state Standard-SHR matched count from the real instruction",
        "reconstruction": "16x16 four-neighbor Dirichlet harmonic beta=0 gamma=1",
        "guidance": "lambda=0.5 dimensions 0..5; gripper clean",
        "closed_loop_scenes": {task: len(seed_list) for task, seed_list in scenes.items()},
        "closed_loop_total_scenes": sum(map(len, scenes.values())),
        "closed_loop_total_six_arm_episodes": 6 * sum(map(len, scenes.values())),
        "reused_arms": ["vanilla", "correct"],
    }
    atomic_json(ARTIFACT / "CONFIG_LOCK.json", lock)
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
