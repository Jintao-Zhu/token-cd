"""A/B compare two guided-decode configurations recorded by pi0_fast_batch_validate.py.

Answers the question that decides whether an optimization may be adopted: does it change the
FAST token stream or the action chunk? Timing alone is not evidence.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path("/home/leju-suzhou/zjt_ws/token-cd")
OUT = REPO / "artifacts/pi0_fast_batch_ab_v1"


def main() -> None:
    a_tag, b_tag = (sys.argv[1:3] + ["base", "batch"])[:2]
    a = json.loads((OUT / f"RUN_{a_tag}.json").read_text())
    b = json.loads((OUT / f"RUN_{b_tag}.json").read_text())
    print(f"[cmp] A={a_tag} (batch={a['batch_branches']})  B={b_tag} (batch={b['batch_branches']})")

    ra = {r["task"]: r for r in a["rows"]}
    rb = {r["task"]: r for r in b["rows"]}
    tasks = [t for t in ra if t in rb]

    # drop the first task in each run from timing (cold, includes compilation)
    ta = [ra[t]["seconds"] for t in tasks[1:]]
    tb = [rb[t]["seconds"] for t in tasks[1:]]

    rows = []
    for t in tasks:
        x, y = ra[t], rb[t]
        same_tok = x["tokens"] == y["tokens"]
        first_div = next((i for i, (p, q) in enumerate(zip(x["tokens"], y["tokens"])) if p != q), -1)
        ax, ay = x["actions"], y["actions"]
        same_shape = len(ax) == len(ay) and len(ax[0]) == len(ay[0])
        maxdiff = max(
            (abs(p - q) for rx, ry in zip(ax, ay) for p, q in zip(rx, ry)), default=float("nan")
        ) if same_shape else float("nan")
        rows.append({
            "task": t,
            "seconds_a": x["seconds"], "seconds_b": y["seconds"],
            "saved_s": x["seconds"] - y["seconds"],
            "tokens_identical": same_tok, "first_token_divergence": first_div,
            "payload_a": x["payload_chars"], "payload_b": y["payload_chars"],
            "stream_valid_a": x["stream_valid"], "stream_valid_b": y["stream_valid"],
            "m_t_a": x["matched_m_t"], "m_t_b": y["matched_m_t"],
            "selected_identical": x["selected_token_ids"] == y["selected_token_ids"],
            "clean_top_a": x["first_clean_top_token"], "clean_top_b": y["first_clean_top_token"],
            "neg_top_a": x["first_negative_top_token"], "neg_top_b": y["first_negative_top_token"],
            "guided_top_a": x["first_guided_top_token"], "guided_top_b": y["first_guided_top_token"],
            "residual_a": x["logit_residual_norm"], "residual_b": y["logit_residual_norm"],
            "action_maxdiff": maxdiff,
            "actions_identical": bool(same_shape and maxdiff == 0.0),
            "action_absmax_a": x["action_absmax"], "action_absmax_b": y["action_absmax"],
        })

    n = len(rows)
    identical_tokens = sum(r["tokens_identical"] for r in rows)
    identical_actions = sum(r["actions_identical"] for r in rows)
    same_m = sum(r["m_t_a"] == r["m_t_b"] for r in rows)
    same_sel = sum(r["selected_identical"] for r in rows)
    summary = {
        "protocol": "PI0_FAST_BATCH_AB_COMPARE_V1",
        "config_a": a_tag, "config_b": b_tag,
        "batch_a": a["batch_branches"], "batch_b": b["batch_branches"],
        "states": n,
        "mean_seconds_a": statistics.fmean(ta), "mean_seconds_b": statistics.fmean(tb),
        "mean_saved_s": statistics.fmean(x - y for x, y in zip(ta, tb)),
        "mean_seconds_a_incl_cold": a["mean_seconds"], "mean_seconds_b_incl_cold": b["mean_seconds"],
        "tokens_identical": identical_tokens,
        "actions_identical": identical_actions,
        "m_t_identical": same_m,
        "selected_identical": same_sel,
        "max_action_maxdiff": max(r["action_maxdiff"] for r in rows if r["action_maxdiff"] == r["action_maxdiff"]),
        "all_streams_valid_a": all(r["stream_valid_a"] for r in rows),
        "all_streams_valid_b": all(r["stream_valid_b"] for r in rows),
        "verdict": (
            "EQUIVALENT" if identical_tokens == n and identical_actions == n
            else "DIVERGES"
        ),
        "rows": rows,
    }
    (OUT / f"COMPARE_{a_tag}_vs_{b_tag}.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"[cmp] steady mean: A={summary['mean_seconds_a']:.2f}s  B={summary['mean_seconds_b']:.2f}s  "
          f"saved={summary['mean_saved_s']:.2f}s "
          f"({100*summary['mean_saved_s']/summary['mean_seconds_a']:.1f}%)")
    print(f"[cmp] tokens identical:  {identical_tokens}/{n}")
    print(f"[cmp] actions identical: {identical_actions}/{n}  max|Δ|={summary['max_action_maxdiff']:.3e}")
    print(f"[cmp] m_t identical: {same_m}/{n}   selected set identical: {same_sel}/{n}")
    print(f"[cmp] streams valid: A={summary['all_streams_valid_a']} B={summary['all_streams_valid_b']}")
    print(f"[cmp] VERDICT: {summary['verdict']}")
    for r in rows:
        flag = "" if r["tokens_identical"] else f"  <-- first token div at {r['first_token_divergence']}"
        print(f"[cmp]   {r['task'][:34]:34s} {r['seconds_a']:6.2f}->{r['seconds_b']:6.2f}s "
              f"save={r['saved_s']:5.2f}s tok_eq={r['tokens_identical']} act_eq={r['actions_identical']}{flag}")


if __name__ == "__main__":
    main()
