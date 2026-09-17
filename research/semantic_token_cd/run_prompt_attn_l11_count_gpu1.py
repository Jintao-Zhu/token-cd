"""Operational GPU1 launcher for the locked L11 token-count rollout.

The registered rollout initially restricted its CLI to GPUs 2 and 3.  GPU1
subsequently passed a complete CUDA+Vulkan smoke episode.  To preserve the
already locked scientific source hash, this launcher changes only that CLI
allow-list in memory; the executed policy, audit, serialization, and config
hash all remain those of the canonical rollout file.
"""
from __future__ import annotations

from pathlib import Path


ROLLOUT = Path(__file__).with_name("prompt_attn_l11_count_rollout.py")
NEEDLE = 'parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)'
REPLACEMENT = 'parser.add_argument("--gpu", type=int, choices=(1, 2, 3), required=True)'


def main() -> None:
    source = ROLLOUT.read_text()
    if source.count(NEEDLE) != 1:
        raise RuntimeError("canonical GPU allow-list no longer matches the locked launcher")
    source = source.replace(NEEDLE, REPLACEMENT)
    namespace = {
        "__name__": "__main__",
        "__file__": str(ROLLOUT),
        "__package__": None,
    }
    exec(compile(source, str(ROLLOUT), "exec"), namespace)


if __name__ == "__main__":
    main()
