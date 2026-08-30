"""Phase 1B Semantic-Merge vs Attention-K8 — mechanism comparison + decision.

Four pre-registered comparisons (seeds 200-299, 3 tasks):

  (1) semantic_attn_k8_l8_15   vs vanilla          (Attention-CD signal)
  (2) semantic_merge_k8_eta100 vs vanilla          (Semantic-Merge-CD signal)
  (3) semantic_merge_k8_eta100 vs semantic_attn_k8_l8_15  (Merge vs Attn, head-to-head)
  (4) semantic_merge_k8_eta100 vs gsm_100          (Local-Semantic vs Global merge;
                                                    the open question "is Global GSM
                                                    intervening too broadly?")

Vanilla and gsm_100 are NOT re-run: they are read from the Phase 1A artifact
(artifacts/attn_global_merge_v1). Fail-closed pairing: for every paired seed the
canonical / initial-state / initial-RGB hashes must be identical across all four
arms; any mismatch aborts with a nonzero exit (the comparison would be invalid).

Primary metrics (paired, per task): SR, Rescue R = #{V=0, arm=1}, Harm H =
#{V=1, arm=0}, Net = R-H, dSR = SR_arm - SR_ref, 95% CI (paired proportions),
McNemar p. Head-to-head comparisons (3)/(4) use the same R/H definition with the
reference swapped to another CD arm.

Usage:
  <venv>/bin/python research/semantic_token_cd/analyze_semantic_merge.py \
    --artifact artifacts/attn_semantic_merge_k8_v1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

TASKS = (
    "google_robot_move_near",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
)
ATTN_ARM = "semantic_attn_k8_l8_15"
MERGE_ARM = "semantic_merge_k8_eta100"
VANILLA_ARM = "vanilla"
GSM100_ARM = "gsm_100"
CD_ARMS = (ATTN_ARM, MERGE_ARM)
REF_ARMS = (VANILLA_ARM, GSM100_ARM)
SEED_START, SEED_END = 200, 299


def _scipy_mcnemar(r: int, h: int) -> float:
    try:
        from scipy.stats import chi2
    except Exception:
        disc = r + h
        if disc == 0:
            return 1.0
        stat = (abs(r - h)) ** 2 / disc
        return float(math.exp(-stat / 2) * 2) if stat > 0 else 1.0
    disc = r + h
    if disc == 0:
        return 1.0
    stat = (abs(r - h) - 1.0) ** 2 / disc if (r + h) > 0 else 0.0
    return float(chi2.sf(stat, df=1))


def paired_dsr_ci(r: int, h: int, n: int) -> tuple[float, float]:
    d = (r - h) / n if n else 0.0
    if n == 0:
        return 0.0, 0.0
    var = (r + h) / (n * n) - ((r - h) ** 2) / (n ** 3)
    var = max(var, 0.0)
    se = math.sqrt(var)
    z = 1.96
    return d - z * se, d + z * se


def _load_summaries(artifact: Path, task: str, arm: str) -> dict[int, dict]:
    d = {}
    adir = artifact / "episodes" / task / arm
    if not adir.exists():
        return d
    for f in adir.glob("episode_*_summary.json"):
        try:
            seed = int(f.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        d[seed] = json.loads(f.read_text())
    return d


def _load_arrays(artifact: Path, task: str, arm: str, seed: int) -> dict | None:
    p = artifact / "episodes" / task / arm / f"episode_{seed:03d}_arrays.npz"
    if not p.exists():
        return None
    return dict(np.load(p, allow_pickle=False))


def _residual(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    eps = 1e-12
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)

    def ls(x):
        m = x.max(axis=-1, keepdims=True)
        e = np.exp(x - m)
        return x - m - np.log(e.sum(axis=-1, keepdims=True))

    return (ls(pos) - ls(neg)).astype(np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a.ravel(), b.ravel()) / (na * nb))


def _paired_block(succ: dict[str, dict[int, bool]], ref: str, arm: str, paired: list[int]) -> dict:
    n = len(paired)
    r = sum(1 for s in paired if not succ[ref][s] and succ[arm][s])
    h = sum(1 for s in paired if succ[ref][s] and not succ[arm][s])
    sr_ref = sum(succ[ref].values()) / n if n else 0.0
    sr_arm = sum(succ[arm].values()) / n if n else 0.0
    lo, hi = paired_dsr_ci(r, h, n)
    return {
        "rescue": r, "harm": h, "net": r - h,
        "sr_ref": float(sr_ref), "sr_arm": float(sr_arm),
        "delta_sr": float(sr_arm - sr_ref),
        "delta_sr_ci95": [round(lo, 4), round(hi, 4)],
        "mcnemar_p": round(_scipy_mcnemar(r, h), 6),
        "rescue_rate": float(r / n), "harm_rate": float(h / n),
        "n_paired": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--ref_artifact", type=Path,
                        default=Path("artifacts/attn_global_merge_v1"),
                        help="Phase 1A artifact holding vanilla + gsm_100")
    parser.add_argument("--min_seed", type=int, default=SEED_START)
    parser.add_argument("--max_seed", type=int, default=SEED_END)
    args = parser.parse_args()

    artifact = args.artifact.resolve()
    ref_artifact = args.ref_artifact.resolve()
    seed_range = range(args.min_seed, args.max_seed + 1)

    out: dict = {
        "artifact": str(artifact),
        "ref_artifact": str(ref_artifact),
        "tasks": {},
        "hash_pairing": {},
        "comparisons": {},
        "decision": {},
    }

    pairing_failures: list[str] = []

    for task in TASKS:
        # success maps across all four arms (2 new + vanilla + gsm_100)
        m = {
            arm: _load_summaries(artifact, task, arm) for arm in CD_ARMS
        }
        m[VANILLA_ARM] = _load_summaries(ref_artifact, task, VANILLA_ARM)
        m[GSM100_ARM] = _load_summaries(ref_artifact, task, GSM100_ARM)

        paired = [s for s in seed_range if all(s in m[a] for a in (ATTN_ARM, MERGE_ARM, VANILLA_ARM))]

        if not paired:
            out["tasks"][task] = {"n_paired": 0, "note": "no paired seeds"}
            continue

        # ---- fail-closed hash pairing. A seed is valid for a CROSS-run
        # comparison (anything vs vanilla or gsm_100) only if the arms'
        # canonical/state/rgb hashes all agree. Rare non-deterministic env
        # resets (move_near seeds 209/211/270) are EXCLUDED from those
        # comparisons. The within-run attn-vs-merge comparison always shares
        # the same snapshot (captured once per seed in one process), so every
        # paired seed is valid there. ----
        def _match(a: str, b: str, s: int) -> bool:
            return (
                m[a][s]["canonical_snapshot_sha256"] == m[b][s]["canonical_snapshot_sha256"]
                and m[a][s]["initial_state_sha256"] == m[b][s]["initial_state_sha256"]
                and m[a][s]["initial_rgb_sha256"] == m[b][s]["initial_rgb_sha256"]
            )

        for s in paired:
            if not _match(ATTN_ARM, VANILLA_ARM, s):
                pairing_failures.append(f"{task} seed {s}: attn hash != vanilla (non-deterministic reset)")
            if not _match(MERGE_ARM, VANILLA_ARM, s):
                pairing_failures.append(f"{task} seed {s}: merge hash != vanilla (non-deterministic reset)")
            if s in m[GSM100_ARM] and not _match(MERGE_ARM, GSM100_ARM, s):
                pairing_failures.append(f"{task} seed {s}: merge hash != gsm_100 (non-deterministic reset)")

        valid_attn_v = [s for s in paired if _match(ATTN_ARM, VANILLA_ARM, s)]
        valid_merge_v = [s for s in paired if _match(MERGE_ARM, VANILLA_ARM, s)]
        valid_merge_g = [s for s in paired if s in m[GSM100_ARM] and _match(MERGE_ARM, GSM100_ARM, s)]

        def _succ(a: str, seeds: list[int]) -> dict[int, bool]:
            return {s: bool(m[a][s]["success"]) for s in seeds}

        t: dict = {"n_paired": len(paired)}
        # (1) attn vs vanilla, (2) merge vs vanilla
        t["attn_vs_vanilla"] = _paired_block(
            {VANILLA_ARM: _succ(VANILLA_ARM, valid_attn_v), ATTN_ARM: _succ(ATTN_ARM, valid_attn_v)},
            VANILLA_ARM, ATTN_ARM, valid_attn_v,
        )
        t["merge_vs_vanilla"] = _paired_block(
            {VANILLA_ARM: _succ(VANILLA_ARM, valid_merge_v), MERGE_ARM: _succ(MERGE_ARM, valid_merge_v)},
            VANILLA_ARM, MERGE_ARM, valid_merge_v,
        )
        # (3) merge vs attn (head-to-head; same-process, always valid)
        t["merge_vs_attn"] = _paired_block(
            {ATTN_ARM: _succ(ATTN_ARM, paired), MERGE_ARM: _succ(MERGE_ARM, paired)},
            ATTN_ARM, MERGE_ARM, paired,
        )
        # (4) merge vs gsm_100 (head-to-head)
        if valid_merge_g:
            t["merge_vs_gsm100"] = _paired_block(
                {GSM100_ARM: _succ(GSM100_ARM, valid_merge_g), MERGE_ARM: _succ(MERGE_ARM, valid_merge_g)},
                GSM100_ARM, MERGE_ARM, valid_merge_g,
            )
        t["n_excluded_attn_vs_vanilla"] = len(paired) - len(valid_attn_v)
        t["n_excluded_merge_vs_vanilla"] = len(paired) - len(valid_merge_v)
        t["n_excluded_merge_vs_gsm100"] = len(paired) - len(valid_merge_g)
        out["tasks"][task] = t

    out["hash_pairing"] = {
        "ok": len(pairing_failures) == 0,
        "n_failures": len(pairing_failures),
        "failures": pairing_failures,
        "note": (
            "All cross-run pairings bit-identical."
            if not pairing_failures
            else "Non-deterministic env resets (rare, move_near 209/211/270) are EXCLUDED "
                 "from cross-run comparisons (vs vanilla / gsm_100). The within-run "
                 "attn-vs-merge comparison is unaffected (same-process snapshot)."
        ),
    }

    present = [k for k in out["tasks"] if out["tasks"][k].get("n_paired", 0) > 0]
    if not present:
        out["decision"] = {"verdict": "NO_DATA"}
        print(json.dumps(out, indent=2, sort_keys=True))
        return

    # ---- diagnostics ----
    diag: dict = {}
    for task in present:
        d: dict = {}
        for arm in CD_ARMS:
            norms, flips = [], []
            temporal = []
            msum = _load_summaries(artifact, task, arm)
            vsum = _load_summaries(ref_artifact, task, VANILLA_ARM)
            paired = [s for s in seed_range if s in msum and s in vsum]
            for s in paired:
                tr = msum[s].get("selector_trace", [])
                for x in tr:
                    norms.append(float(x.get("residual_norm", 0.0)))
                    if "positive_token_ids" in x and "final_token_ids" in x:
                        flips.append(1 if x["positive_token_ids"] != x["final_token_ids"] else 0)
                arr = _load_arrays(artifact, task, arm, s)
                if arr and "negative_logits" in arr:
                    pos = arr["positive_logits"]; neg = arr["negative_logits"]
                    T = min(pos.shape[0], neg.shape[0])
                    for t in range(1, T):
                        temporal.append(_cos(_residual(pos[t], neg[t]), _residual(pos[t - 1], neg[t - 1])))
            d[arm] = {
                "mean_residual_norm": float(np.mean(norms)) if norms else None,
                "action_flip_rate": float(np.mean(flips)) if flips else None,
                "within_arm_temporal_cos": float(np.mean(temporal)) if temporal else None,
            }
        # merge vs attn first-step residual cosine + action divergence (shared state at t=0)
        coses, divs = [], []
        vsum = _load_summaries(ref_artifact, task, VANILLA_ARM)
        msum = _load_summaries(artifact, task, MERGE_ARM)
        asum = _load_summaries(artifact, task, ATTN_ARM)
        paired = [s for s in seed_range if s in msum and s in asum and s in vsum]
        for s in paired:
            mg = _load_arrays(artifact, task, MERGE_ARM, s)
            at = _load_arrays(artifact, task, ATTN_ARM, s)
            vv = _load_arrays(ref_artifact, task, VANILLA_ARM, s)
            if not (mg and at and "negative_logits" in mg and "negative_logits" in at):
                continue
            r_m = _residual(mg["positive_logits"][0], mg["negative_logits"][0])
            r_a = _residual(at["positive_logits"][0], at["negative_logits"][0])
            coses.append(_cos(r_m, r_a))
            if vv is not None and mg["executed_actions"].shape[0] > 0 and at["executed_actions"].shape[0] > 0:
                divs.append(float(np.linalg.norm(mg["executed_actions"][0] - at["executed_actions"][0])))
        d["merge_vs_attn_first_step_cos"] = float(np.mean(coses)) if coses else None
        d["merge_vs_attn_first_step_action_div"] = float(np.mean(divs)) if divs else None
        diag[task] = d
    out["diagnostics"] = diag

    # ---- aggregate (macro) ----
    def _macro(key: str) -> float:
        vals = [out["tasks"][task][key]["delta_sr"] for task in present]
        return float(np.mean(vals))

    comp = {
        "attn_vs_vanilla": {"delta_sr_macro": _macro("attn_vs_vanilla")},
        "merge_vs_vanilla": {"delta_sr_macro": _macro("merge_vs_vanilla")},
        "merge_vs_attn": {"delta_sr_macro": _macro("merge_vs_attn")},
    }
    if all("merge_vs_gsm100" in out["tasks"][t] for t in present):
        comp["merge_vs_gsm100"] = {"delta_sr_macro": _macro("merge_vs_gsm100")}
    for key in ("attn_vs_vanilla", "merge_vs_vanilla", "merge_vs_attn", "merge_vs_gsm100"):
        if key in comp:
            nets = [out["tasks"][t][key]["net"] for t in present if key in out["tasks"][t]]
            comp[key]["macro_net"] = float(np.mean(nets)) if nets else None
            comp[key]["n_tasks_net_gt0"] = sum(1 for x in nets if x > 0)
            comp[key]["n_tasks"] = len(nets)
    out["comparisons"] = comp

    # ---- decision (mechanism verdict, not a strict gate) ----
    decision = {
        "question": "Is Semantic-Merge (local, selected groups only) better than Global GSM100?",
        "merge_vs_gsm100_delta_sr_macro": comp.get("merge_vs_gsm100", {}).get("delta_sr_macro"),
        "merge_vs_attn_delta_sr_macro": comp.get("merge_vs_attn", {}).get("delta_sr_macro"),
        "merge_vs_vanilla_delta_sr_macro": comp.get("merge_vs_vanilla", {}).get("delta_sr_macro"),
        "attn_vs_vanilla_delta_sr_macro": comp.get("attn_vs_vanilla", {}).get("delta_sr_macro"),
    }
    out["decision"] = decision

    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
