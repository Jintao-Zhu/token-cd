"""Is the FAST payload character count exactly the number of UTF-8 lead bytes?

Hypothesis: the FAST tokenizer is byte-level BPE, so each code maps to a fixed BYTE string and
`decode(codes)` is utf-8 decoding of the concatenated bytes with errors='replace'. If so, the
number of decoded characters equals the number of bytes that are NOT UTF-8 continuation bytes
((b & 0xC0) != 0x80), because utf-8 is self-synchronising and every invalid byte yields one
U+FFFD. That quantity is ADDITIVE over codes, which would make the decode loop's length
accounting exact with a one-line table change instead of a per-candidate tokenizer call.

Verified here exhaustively against the real tokenizer on random sequences of length 1..40,
including sequences built only from the codes that individually decode to U+FFFD.
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


def bytes_to_unicode():
    """GPT-2 byte-level alphabet, as used by byte-level BPE tokenizers."""
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

    # per-code byte string from the byte-level token spelling
    code_bytes: list[bytes | None] = []
    for c in range(vocab_size):
        ts = bpe.convert_ids_to_tokens(c)
        if ts is None:
            code_bytes.append(None)
            continue
        try:
            code_bytes.append(bytes(b2u[ch] for ch in ts))
        except KeyError:
            code_bytes.append(None)
    n_ok = sum(1 for x in code_bytes if x is not None)
    print(f"[lead] vocab={vocab_size} codes with a byte-level spelling: {n_ok}", flush=True)

    lead = np.asarray(
        [sum(1 for b in (x or b"") if (b & 0xC0) != 0x80) for x in code_bytes], dtype=np.int64
    )
    naive = np.asarray([len(bpe.decode([c])) for c in range(vocab_size)], dtype=np.int64)
    n_diff = int(np.count_nonzero(lead != naive))
    print(f"[lead] codes where lead-byte count != len(decode([c])): {n_diff}", flush=True)

    rng = np.random.default_rng(7)
    ends_bad = [c for c in range(vocab_size) if bpe.decode([c]).endswith(REPL)]
    starts_bad = [c for c in range(vocab_size) if bpe.decode([c]).startswith(REPL)]
    pool_bad = np.asarray(sorted(set(ends_bad) | set(starts_bad)), dtype=np.int64)

    mismatches = []
    tested = 0
    for length in (1, 2, 3, 5, 10, 26, 40):
        for trial in range(3000):
            # half the trials drawn only from the problematic codes, to stress the merges
            if trial % 2 == 0 and pool_bad.size:
                seq = rng.choice(pool_bad, size=length)
            else:
                seq = rng.integers(0, vocab_size, size=length)
            real = len(bpe.decode([int(x) for x in seq]))
            pred = int(lead[seq].sum())
            tested += 1
            if real != pred:
                mismatches.append({
                    "codes": [int(x) for x in seq], "real": real, "predicted": pred,
                    "naive": int(naive[seq].sum()),
                })
    print(f"[lead] sequences tested={tested}  mismatches={len(mismatches)}", flush=True)
    if mismatches:
        for m in mismatches[:5]:
            print(f"       {m}", flush=True)

    # how often would the OLD naive table have been wrong?
    naive_wrong = 0
    for _ in range(20000):
        seq = rng.integers(0, vocab_size, size=26)
        if len(bpe.decode([int(x) for x in seq])) != int(naive[seq].sum()):
            naive_wrong += 1
    print(f"[lead] 26-token streams where the OLD naive sum is wrong: {naive_wrong}/20000 "
          f"({100*naive_wrong/20000:.3f}%)", flush=True)

    summary = {
        "protocol": "PI0_FAST_LEAD_BYTE_LENGTH_V1",
        "vocab_size": vocab_size,
        "codes_with_byte_spelling": n_ok,
        "codes_where_lead_differs_from_naive": n_diff,
        "sequences_tested": tested,
        "lead_byte_mismatches": len(mismatches),
        "naive_wrong_on_26_token_streams": naive_wrong,
        "naive_wrong_rate_26": naive_wrong / 20000,
        "verdict": "lead-byte count is EXACT" if not mismatches else "lead-byte count is NOT exact",
        "lead_lengths": lead.tolist(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "LEAD_BYTE_LENGTH.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[lead] {summary['verdict']}", flush=True)


if __name__ == "__main__":
    main()
