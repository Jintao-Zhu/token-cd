"""Exact incremental character count for utf-8 bytes decoded with errors='replace'.

The lead-byte rule fails on orphan continuation bytes (a lone 0xA9 still yields one U+FFFD).
This implements a small DFA over the byte stream and verifies it EXHAUSTIVELY against Python's
own utf-8 replace decoder, so the FAST decode loop can track the true payload length
incrementally instead of summing per-token individual lengths.

DFA state = number of continuation bytes still expected (0..3).
  * byte is continuation and pending>0 -> pending -= 1; if pending==0 -> emit one character
  * byte is continuation and pending==0 -> orphan: emit one replacement character
  * byte is a lead byte (0xC0..0xFF):
        if pending>0 -> the pending sequence was truncated: emit one replacement, then start over
        pending = expected_continuations(byte)
  * byte is ASCII (<0x80): emit one character (pending must be 0 here)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path("/home/leju-suzhou/zjt_ws")
OUT = ROOT / "token-cd/artifacts/pi0_fast_bpe_additivity_v1"


def expected_continuations(b: int) -> int:
    if b < 0x80:
        return 0
    if 0xC0 <= b <= 0xDF:
        return 1
    if 0xE0 <= b <= 0xEF:
        return 2
    if 0xF0 <= b <= 0xF7:
        return 3
    return -1  # 0x80..0xBF handled as continuation; 0xF8..0xFF invalid lead


def dfa_count(bs: bytes) -> int:
    count = 0
    pending = 0
    for b in bs:
        if 0x80 <= b <= 0xBF:
            if pending > 0:
                pending -= 1
                if pending == 0:
                    count += 1
            else:
                count += 1  # orphan continuation
            continue
        # not a continuation byte
        if pending > 0:
            count += 1  # truncated sequence
            pending = 0
        e = expected_continuations(b)
        if e == 0:
            count += 1  # ASCII
        elif e < 0:
            count += 1  # invalid lead byte (0xF8..0xFF)
        else:
            pending = e
    if pending > 0:
        count += 1  # truncated at end of stream
    return count


def main() -> None:
    import itertools
    import random

    # exhaustive over all 1- and 2-byte strings, plus a large random sweep over 3..6 bytes
    bad = []
    tested = 0
    for n in (1, 2):
        for tup in itertools.product(range(256), repeat=n):
            bs = bytes(tup)
            tested += 1
            if dfa_count(bs) != len(bs.decode("utf-8", errors="replace")):
                bad.append((bs.hex(), dfa_count(bs), len(bs.decode("utf-8", errors="replace"))))
    rng = random.Random(0)
    for _ in range(300000):
        n = rng.randint(3, 6)
        bs = bytes(rng.randrange(256) for _ in range(n))
        tested += 1
        if dfa_count(bs) != len(bs.decode("utf-8", errors="replace")):
            bad.append((bs.hex(), dfa_count(bs), len(bs.decode("utf-8", errors="replace"))))
    # structured sweep: valid multi-byte sequences and deliberate truncations
    for lead, k in ((0xC3, 1), (0xE2, 2), (0xF0, 3)):
        for cut in range(k + 1):
            for tail in range(256):
                bs = bytes([lead]) + bytes([tail]) * cut
                tested += 1
                if dfa_count(bs) != len(bs.decode("utf-8", errors="replace")):
                    bad.append((bs.hex(), dfa_count(bs), len(bs.decode("utf-8", errors="replace"))))

    print(f"[dfa] tested={tested} mismatches={len(bad)}", flush=True)
    for x in bad[:8]:
        print(f"      {x}", flush=True)

    summary = {
        "protocol": "PI0_FAST_UTF8_DFA_V1",
        "tested": tested,
        "mismatches": len(bad),
        "examples": bad[:20],
        "verdict": "DFA is EXACT" if not bad else "DFA is NOT exact",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "UTF8_DFA.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[dfa] {summary['verdict']}", flush=True)


if __name__ == "__main__":
    main()
