"""Build the frozen episode-internal permutation tape for the state-coupling ablation.

For every (task, seed) the ORIGINAL matched rollout supplies one integer per
control step.  We keep that multiset exactly and only permute its order, so the
shuffle arm gets an identical episode-level budget distribution (mean, std,
min/max, histogram, value set) but the m_t <-> x_t correspondence is destroyed.

The permutation is generated from a per-(task, seed) derived RNG so the tape is
reproducible and independent of iteration order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
SEEDS = tuple(range(100, 200))
BASE_RNG_SEED = 20260918


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched-artifact", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    matched = args.matched_artifact.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    tapes: dict[str, dict[str, list[int]]] = {}
    diag: dict[str, dict[str, dict]] = {}
    pool: dict[str, list[int]] = {}
    agg = {k: [] for k in ("index_displacement", "exact_equal", "mean_abs_delta", "pearson")}

    for ti, task in enumerate(TASKS):
        tapes[task] = {}
        diag[task] = {}
        pool[task] = []
        for seed in SEEDS:
            src = matched / "episodes" / task / "l11_matched" / f"episode_{seed:03d}_summary.json"
            d = json.loads(src.read_text())
            m = [int(y["actual_selected_count"]) for y in d["selector_trace"]
                 if y.get("actual_selected_count") is not None]
            if not m:
                raise RuntimeError(f"empty matched trace: {task}/{seed}")
            L = len(m)
            rng = np.random.default_rng([BASE_RNG_SEED, ti, seed])
            perm = rng.permutation(L)
            shuffled = [m[i] for i in perm]
            # diagnostics
            idx_disp = float(np.mean(perm != np.arange(L)))
            exact = float(np.mean(np.asarray(shuffled) == np.asarray(m)))
            mad = float(np.mean(np.abs(np.asarray(shuffled) - np.asarray(m))))
            corr = float(np.corrcoef(np.asarray(m), np.asarray(shuffled))[0, 1]) if L > 1 else float("nan")
            tapes[task][str(seed)] = shuffled
            diag[task][str(seed)] = {
                "length": L,
                "index_displacement_rate": idx_disp,
                "exact_m_equality_rate": exact,
                "mean_abs_delta": mad,
                "pearson_with_matched": corr,
            }
            agg["index_displacement"].append(idx_disp)
            agg["exact_equal"].append(exact)
            agg["mean_abs_delta"].append(mad)
            agg["pearson"].append(corr)
            pool[task].extend(m)

    tape_path = out / "SHUFFLE_TAPES.json"
    payload = {
        "protocol_id": "PROMPT_ATTN_L11_MATCHED_STATE_COUPLING_V1",
        "note": "episode-internal permutation of the original matched budget tape; multiset preserved exactly",
        "base_rng_seed": BASE_RNG_SEED,
        "rng_key": "[base_rng_seed, task_index, seed]",
        "tasks": list(TASKS),
        "seeds": list(SEEDS),
        "source_artifact": str(matched),
        "tapes": tapes,
        "extension_pool": pool,
        "diagnostics": diag,
        "aggregate": {k: float(np.mean(v)) for k, v in agg.items()},
        "sha256": "",
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps({"tapes": tapes, "seed": BASE_RNG_SEED}, sort_keys=True).encode()
    ).hexdigest()
    tape_path.write_text(json.dumps(payload, indent=1) + "\n")

    print(json.dumps({
        "tape_file": str(tape_path),
        "tasks": len(TASKS),
        "seeds_per_task": len(SEEDS),
        "aggregate_diagnostics": payload["aggregate"],
        "sha256": payload["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
