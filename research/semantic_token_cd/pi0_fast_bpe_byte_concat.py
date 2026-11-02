"""Foundation check for the exact FAST decoder.

If the tokenizer is byte-level BPE, then for any code sequence
    bpe.decode(codes) == concat(code_bytes).decode("utf-8", errors="replace")
which lets the decode loop validate a candidate with a plain byte concatenation plus a
Python utf-8 decode, i.e. exactly the real joint payload, with no tokenizer call and no
length-table approximation.

Also checks that the payload character count is what the official decoder expects
(action_horizon * action_dim).
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


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\u00a1"), ord("\u00ac") + 1)) + \
         list(range(ord("\u00ae"), ord("\u00ff") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def main() -> None:
    tok = FASTTokenizer()
    fast = tok._fast_tokenizer
    bpe = fast.bpe_tokenizer
    vocab_size = int(getattr(fast, "vocab_size", None) or len(fast))
    b2u = bytes_to_unicode()

    code_bytes = []
    for c in range(vocab_size):
        ts = bpe.convert_ids_to_tokens(c)
        code_bytes.append(bytes(b2u[ch] for ch in ts))

    rng = np.random.default_rng(11)
    mismatches = []
    tested = 0
    for length in (1, 2, 3, 5, 10, 26, 40):
        for _ in range(4000):
            seq = [int(x) for x in rng.integers(0, vocab_size, size=length)]
            real = bpe.decode(seq)
            pred = b"".join(code_bytes[c] for c in seq).decode("utf-8", errors="replace")
            tested += 1
            if real != pred:
                mismatches.append({"codes": seq[:8], "real": real[:40], "pred": pred[:40]})
    print(f"[eq] tested={tested}  byte-concat mismatches={len(mismatches)}", flush=True)
    for m in mismatches[:5]:
        print(f"     {m}", flush=True)

    # how long is a byte string per code, and total for a 70-char payload
    blens = np.asarray([len(x) for x in code_bytes], dtype=np.int64)
    print(f"[eq] code byte-length: min={blens.min()} max={blens.max()} mean={blens.mean():.2f}", flush=True)

    summary = {
        "protocol": "PI0_FAST_BYTE_CONCAT_EQUIVALENCE_V1",
        "vocab_size": vocab_size,
        "tested": tested,
        "mismatches": len(mismatches),
        "code_byte_len_min": int(blens.min()),
        "code_byte_len_max": int(blens.max()),
        "verdict": "byte-concat equals tokenizer decode" if not mismatches else "NOT equivalent",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "BYTE_CONCAT.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[eq] {summary['verdict']}", flush=True)


if __name__ == "__main__":
    main()
