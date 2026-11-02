#!/usr/bin/env python3
"""Re-analyse the EARLIER lambda=0.5 budget experiments with the same pairing method used now.

Compare, on identical seeds and tasks:
  * artifacts/prompt_attn_l11_token_count_v1  (l11_matched vs l11_k16/24/32/48/64, seeds 100-199)
  * artifacts/prompt_attn_l11_top_p_v1        (matched vs l11_top_p75/80/85)
  * artifacts/prompt_attn_l11_top_p90_extension_v1 (l11_top_p90)

Reads only the two scalar fields it needs (regex on raw bytes) because the summaries carry a
multi-hundred-KB trace.
"""
from __future__ import annotations

import collections
import math
import os
import re
from pathlib import Path

STUDIES = {
    "count": Path("artifacts/prompt_attn_l11_token_count_v1"),
    "top_p": Path("artifacts/prompt_attn_l11_top_p_v1"),
    "top_p90": Path("artifacts/prompt_attn_l11_top_p90_extension_v1"),
}
SEED_RE = re.compile(rb'"seed": (\d+)')
SUCC_RE = re.compile(rb'"success": (true|false)')
TASKS = ["google_robot_close_drawer", "google_robot_open_drawer",
         "google_robot_move_near", "google_robot_pick_coke_can"]


def mcn(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def read_arm(root: Path, task: str, arm: str) -> dict[int, bool]:
    out = {}
    d = root / "episodes" / task / arm
    if not d.is_dir():
        return out
    for f in os.scandir(d):
        if not f.name.endswith("_summary.json"):
            continue
        data = open(f.path, "rb").read()
        m1, m2 = SEED_RE.search(data), SUCC_RE.search(data)
        if m1 and m2:
            out[int(m1.group(1))] = m2.group(1) == b"true"
    return out


def main() -> int:
    # ---- lambda=0.5 fixed-count study (matched in the same artifact) ----
    for name, arms in (("count", ["l11_k16", "l11_k24", "l11_k32", "l11_k48", "l11_k64"]),
                       ("top_p", ["l11_top_p75", "l11_top_p80", "l11_top_p85"]),
                       ("top_p90", ["l11_top_p90"])):
        root = STUDIES[name]
        ref_file = root / "episodes" / TASKS[0] / "l11_matched"
        matched_here = ref_file.is_dir()
        if not matched_here:
            alt = STUDIES["count"]
            ref = {(t, s): v for t in TASKS for s, v in read_arm(alt, t, "l11_matched").items()}
            ref_name = "prompt_attn_l11_token_count_v1::l11_matched"
        else:
            ref = {(t, s): v for t in TASKS for s, v in read_arm(root, t, "l11_matched").items()}
            ref_name = f"{name}::l11_matched"
        print(f"\n=== lambda=0.5 · {name}  (matched = {ref_name}, n={len(ref)}) ===")
        print("| arm | 对数 | 该臂 | matched | 净 | 救/坏 | p |")
        print("|---|---|---|---|---|---|---|")
        for a in arms:
            rows = {(t, s): v for t in TASKS for s, v in read_arm(root, t, a).items()}
            p = [(k, v) for k, v in rows.items() if k in ref]
            if not p:
                print(f"| {a} | 0 | — | — | — | — | — |")
                continue
            b = sum(1 for k, v in p if v and not ref[k])
            c = sum(1 for k, v in p if ref[k] and not v)
            sa = sum(1 for k, v in p if v); sr = sum(1 for k, v in p if ref[k])
            print(f"| {a} | {len(p)} | {sa}/{len(p)} | {sr}/{len(p)} | {b-c:+d} | {b}/{c} | {mcn(b,c):.4f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
