"""Run the locked L11 Prompt-Attn policy on canonical seeds 100-299.

The scientific policy/audit implementation is the existing
``prompt_attn_layer_rollout.py``.  This wrapper changes only the CLI seed
guard so the same L11 single-layer arm can be extended from the original
0-99 evaluation to the full canonical 0-299 release.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.semantic_token_cd import prompt_attn_layer_rollout as layer_rollout


def parse_seeds(specification: str) -> list[int]:
    seeds: list[int] = []
    for part in specification.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            seeds.extend(range(lo, hi + 1))
        else:
            seeds.append(int(part))
    result = sorted(set(seeds))
    if not result or any(seed < 0 or seed > 299 for seed in result):
        raise ValueError("full L11 canonical seeds must be within 0..299")
    return result


layer_rollout.parse_seeds = parse_seeds


if __name__ == "__main__":
    layer_rollout.main()
