#!/usr/bin/env python3
"""Episode-cluster summaries for the same-state L11 budget response scan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    frames = [pd.read_json(p, lines=True) for p in sorted(args.input.glob("shard_*.jsonl"))]
    if not frames:
        raise SystemExit("no budget scan shards found")
    data = pd.concat(frames, ignore_index=True)
    key = ["task", "seed", "step", "multiplier"]
    if data.duplicated(key).any():
        dup = data[data.duplicated(key, keep=False)][key].head().to_dict("records")
        raise RuntimeError(f"duplicate state/multiplier rows: {dup}")
    metrics = ["m_used", "D_feat", "D_feat_relative", "D_action_raw", "D_action_scaled_l2",
               "mean_action_logit_js", "action_token_flip_rate", "guided_changed_dims"]
    out = {"protocol_id": "OPENVLA_L11_ABC_BUDGET_DOSE_OFFLINE_V1",
           "rows": int(len(data)), "unique_states": int(data[["task", "seed", "step"]].drop_duplicates().shape[0]),
           "expected_states": 1600, "multipliers": {}, "task_results": {}}
    rng = np.random.default_rng(20261009)
    eps = data[["task", "seed"]].drop_duplicates().to_records(index=False).tolist()
    for multiplier in sorted(data.multiplier.unique()):
        sub = data[data.multiplier == multiplier]
        ep = sub.groupby(["task", "seed"], as_index=False)[metrics].mean()
        record = {"states": int(len(sub)), "episodes": int(len(ep))}
        for metric in metrics:
            vals = ep[metric].to_numpy(float)
            boots = []
            for _ in range(args.bootstrap):
                idx = rng.integers(0, len(vals), len(vals))
                boots.append(float(np.mean(vals[idx])))
            record[metric] = float(np.mean(vals))
            record[metric + "_episode_bootstrap_95ci"] = np.quantile(boots, [0.025, 0.975]).tolist()
        out["multipliers"][str(multiplier)] = record
        out["task_results"][str(multiplier)] = {}
        for task, ts in sub.groupby("task"):
            tep = ts.groupby("seed", as_index=False)[metrics].mean()
            out["task_results"][str(multiplier)][task] = {
                "states": int(len(ts)), "episodes": int(len(tep)),
                **{m: float(tep[m].mean()) for m in metrics},
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(data), "unique_states": out["unique_states"],
                      "expected_states": 1600}, indent=2))


if __name__ == "__main__":
    main()
