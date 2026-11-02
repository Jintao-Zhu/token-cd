"""Why is the FAST payload length not the sum of per-token lengths?

The malformed cases show: 26 code tokens whose precomputed individual lengths sum to 70, but
the joint BPE decode yields 69 characters. The decode loop's termination test accumulates the
individual lengths, so it stops one character late/early and the official decoder then cannot
reshape 69 into (action_horizon, action_dim).

This script measures the actual structure of the non-additivity:
  * does len(decode([u,v])) == len(decode([u])) + len(decode([v])) hold in general?
  * how often does it fail, and for which code ranges?
  * is the tokenizer byte-level (partial UTF-8 sequences) or character-level?
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path("/home/leju-suzhou/zjt_ws")
sys.path.insert(0, str(ROOT / "openpi/src"))

import numpy as np  # noqa: E402

from openpi.models.tokenizer import FASTTokenizer  # noqa: E402

OUT = ROOT / "token-cd/artifacts/pi0_fast_bpe_additivity_v1"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = FASTTokenizer()
    fast = tok._fast_tokenizer
    bpe = fast.bpe_tokenizer
    vocab_size = getattr(fast, "vocab_size", None)
    if vocab_size is None:
        vocab_size = len(fast)
    vocab_size = int(vocab_size)

    print(f"[bpe] fast vocab_size={vocab_size}  action_horizon/dim payload=70", flush=True)
    print(f"[bpe] bpe_tokenizer type={type(bpe).__name__}", flush=True)

    indiv = [len(bpe.decode([c])) for c in range(vocab_size)]
    print(f"[bpe] individual length histogram: {dict(sorted(Counter(indiv).items()))}", flush=True)

    # --- pairwise additivity ---
    rng = np.random.default_rng(0)
    codes = np.arange(vocab_size)
    nonadd_pairs = []
    for _ in range(20000):
        u, v = int(rng.choice(codes)), int(rng.choice(codes))
        joint = len(bpe.decode([u, v]))
        s = indiv[u] + indiv[v]
        if joint != s:
            nonadd_pairs.append((u, v, indiv[u], indiv[v], joint, s))
    print(f"[bpe] pairwise: {len(nonadd_pairs)}/20000 non-additive", flush=True)
    if nonadd_pairs:
        print(f"[bpe] examples (u, v, len_u, len_v, joint, sum):", flush=True)
        for row in nonadd_pairs[:8]:
            print(f"       {row}", flush=True)

    # --- what do the offending individual decodes look like? ---
    if nonadd_pairs:
        u, v = nonadd_pairs[0][0], nonadd_pairs[0][1]
        du, dv, dj = bpe.decode([u]), bpe.decode([v]), bpe.decode([u, v])
        print(f"[bpe] decode([{u}])={du!r} bytes={du.encode('utf-8', 'surrogatepass').hex()}", flush=True)
        print(f"[bpe] decode([{v}])={dv!r} bytes={dv.encode('utf-8', 'surrogatepass').hex()}", flush=True)
        print(f"[bpe] decode([{u},{v}])={dj!r} bytes={dj.encode('utf-8', 'surrogatepass').hex()}", flush=True)

    # --- is it UTF-8 continuation-byte splitting? ---
    def is_cont(ch: str) -> bool:
        try:
            b = ch.encode("utf-8")
        except UnicodeEncodeError:
            return False
        return len(b) == 1 and 0x80 <= b[0] < 0xC0

    n_cont = sum(1 for c in range(vocab_size) if len(bpe.decode([c])) == 1 and is_cont(bpe.decode([c])))
    print(f"[bpe] single tokens decoding to a lone UTF-8 continuation byte: {n_cont}", flush=True)

    summary = {
        "protocol": "PI0_FAST_BPE_ADDITIVITY_V1",
        "vocab_size": vocab_size,
        "individual_length_histogram": {str(k): int(v) for k, v in sorted(Counter(indiv).items())},
        "pairwise_tested": 20000,
        "pairwise_non_additive": len(nonadd_pairs),
        "lone_continuation_byte_tokens": int(n_cont),
        "examples": [list(map(int, r)) for r in nonadd_pairs[:20]],
    }
    (OUT / "ADDITIVITY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[bpe] verdict: non-additivity {'CONFIRMED' if nonadd_pairs else 'not observed'}", flush=True)


if __name__ == "__main__":
    main()
