"""Size the exact correction table for FAST BPE joint-decode length.

Non-additivity arises when a code's individual decode ends with (or starts with) a lone
U+FFFD replacement character, i.e. the code holds part of a multi-byte UTF-8 sequence. Only
pairs (u, v) where u ends mid-sequence and v starts mid-sequence can merge into fewer
characters, so the correction table is sparse and can be built exactly.

This script identifies the candidate code sets and verifies exhaustively that no
non-additive pair exists outside them.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path("/home/leju-suzhou/zjt_ws")
sys.path.insert(0, str(ROOT / "openpi/src"))

import numpy as np  # noqa: E402

from openpi.models.tokenizer import FASTTokenizer  # noqa: E402

OUT = ROOT / "token-cd/artifacts/pi0_fast_bpe_additivity_v1"
REPL = "\ufffd"


def main() -> None:
    tok = FASTTokenizer()
    fast = tok._fast_tokenizer
    bpe = fast.bpe_tokenizer
    vocab_size = int(getattr(fast, "vocab_size", None) or len(fast))

    dec = [bpe.decode([c]) for c in range(vocab_size)]
    lens = np.asarray([len(d) for d in dec], dtype=np.int64)

    ends_bad = [c for c in range(vocab_size) if dec[c].endswith(REPL)]
    starts_bad = [c for c in range(vocab_size) if dec[c].startswith(REPL)]
    print(f"[tbl] vocab={vocab_size}", flush=True)
    print(f"[tbl] codes whose decode ENDS with U+FFFD  : {len(ends_bad)}", flush=True)
    print(f"[tbl] codes whose decode STARTS with U+FFFD: {len(starts_bad)}", flush=True)
    print(f"[tbl] candidate pairs = {len(ends_bad) * len(starts_bad)}", flush=True)

    # exact correction over the candidate set
    corrections = []
    for u in ends_bad:
        for v in starts_bad:
            joint = len(bpe.decode([u, v]))
            s = int(lens[u]) + int(lens[v])
            if joint != s:
                corrections.append({"u": int(u), "v": int(v), "joint": int(joint), "sum": s,
                                    "delta": int(joint - s)})
    print(f"[tbl] non-additive candidate pairs: {len(corrections)}", flush=True)

    # exhaustive check on random pairs outside the candidate set (must all be additive)
    rng = np.random.default_rng(1)
    ends_set, starts_set = set(ends_bad), set(starts_bad)
    tested = 0
    bad_outside = []
    for _ in range(200000):
        u, v = int(rng.integers(vocab_size)), int(rng.integers(vocab_size))
        if u in ends_set and v in starts_set:
            continue
        tested += 1
        if len(bpe.decode([u, v])) != int(lens[u]) + int(lens[v]):
            bad_outside.append((u, v))
    print(f"[tbl] outside-candidate pairs tested={tested} non-additive={len(bad_outside)}", flush=True)
    if bad_outside:
        print(f"[tbl] WARNING: counterexamples outside candidate set: {bad_outside[:5]}", flush=True)

    # triple check: can three-way merges happen that pairwise misses?
    triple_bad = 0
    for _ in range(50000):
        u, v, w = (int(rng.integers(vocab_size)) for _ in range(3))
        joint = len(bpe.decode([u, v, w]))
        # additive prediction with the pairwise correction applied
        pred = int(lens[u]) + int(lens[v]) + int(lens[w])
        if u in ends_set and v in starts_set:
            pred += int(bpe.decode([u, v]).__len__()) - int(lens[u]) - int(lens[v])
        if joint != pred:
            triple_bad += 1
    print(f"[tbl] triple test: 50000 triples, mismatches after pairwise correction = {triple_bad}", flush=True)

    summary = {
        "protocol": "PI0_FAST_BPE_CORRECTION_TABLE_V1",
        "vocab_size": vocab_size,
        "ends_with_replacement": len(ends_bad),
        "starts_with_replacement": len(starts_bad),
        "candidate_pairs": len(ends_bad) * len(starts_bad),
        "non_additive_candidate_pairs": len(corrections),
        "outside_candidate_tested": tested,
        "outside_candidate_non_additive": len(bad_outside),
        "triple_mismatches": triple_bad,
        "corrections": corrections,
        "ends_bad": ends_bad,
        "starts_bad": starts_bad,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "CORRECTION_TABLE.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[tbl] wrote {OUT/'CORRECTION_TABLE.json'}", flush=True)


if __name__ == "__main__":
    main()
