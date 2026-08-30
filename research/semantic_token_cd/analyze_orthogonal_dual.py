#!/usr/bin/env python3
"""ORTHOGONAL_DUAL_ATTENTION_CD_50_V1 — outcome + mechanism analysis.

CPU-only: reads the stored per-episode summaries and the reference three-arm
pairing manifest. Never loads the model, never touches the GPU, never disturbs
the running rollout.

Produces:
  1. Per-task SR for all 4 arms (vanilla / attn_uniform / attn_semantic /
     orthogonal_dual), restricted to the orthogonal_dual seed set
     (denominator-matched via canonical snapshot pairing).
  2. Paired Rescue/Harm for Dual vs Uniform and Dual vs Semantic, with the
     McNemar exact p-value and a paired-bootstrap 95% CI on the SR difference.
  3. Per-task mechanism diagnostics from the stored selector_trace:
        cos(r_sem, r_uniform)            == cos_sim_raw
        ||r_sem^orthogonal|| / ||r_sem|| == ortho_ratio
     (r_geom = pos - uniform, r_sem = pos - semantic,
      r_sem^orthogonal = r_sem - proj_{r_geom}(r_sem)).
  4. Coke's "Dual-only rescue" episodes:
        vanilla fail AND uniform fail AND semantic fail AND dual success.

Run:
  cd /home/leju-suzhou/zjt_ws/token-cd
  task1/.venvs/openvla-ar/bin/python research/semantic_token_cd/analyze_orthogonal_dual.py
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

OD_ART = Path("artifacts/orthogonal_dual_attention_cd_50_v1")
REF_ART = Path("artifacts/uniform_vs_semantic_attention_cd_450_v1")
TASKS = (
    "google_robot_pick_coke_can",
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_move_near",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_stack_cube",
)
REF_ARMS = ("vanilla", "attn_uniform", "attn_semantic")
ARM_LABEL = {"vanilla": "vanilla", "attn_uniform": "uniform", "attn_semantic": "semantic"}
N_BOOT = 20000
BOOT_SEED = 0


def load_od(task: str) -> dict[int, dict]:
    d = OD_ART / "episodes" / task / "orthogonal_dual"
    out = {}
    for f in d.glob("episode_*_summary.json"):
        sm = json.loads(f.read_text())
        out[int(sm["seed"])] = sm
    return out


def load_ref(task: str) -> dict[int, dict[str, bool | None]]:
    m = json.loads((REF_ART / "episodes" / task / "pairing_manifest.json").read_text())
    out: dict[int, dict[str, bool | None]] = {}
    for pair in m["pairs"]:
        s = int(pair["seed"])
        row: dict[str, bool | None] = {}
        for arm in REF_ARMS:
            p = pair["sources"].get(arm)
            row[arm] = (
                json.loads(Path(p).read_text())["success"]
                if p and Path(p).exists()
                else None
            )
        out[s] = row
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p (b = ref fail & dual success, c = ref success & dual fail)."""
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    p_one = sum(comb(n, k) for k in range(m + 1)) * 0.5 ** n
    return min(1.0, 2.0 * p_one)


def paired_bootstrap(a: list[bool], b: list[bool], n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    rng = np.random.RandomState(seed)
    obs = a.mean() - b.mean()
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


def rescue_harm(dual: dict[int, bool], ref: dict[int, bool], seeds: list[int]):
    rescue = harm = 0
    for s in seeds:
        d, r = dual[s], ref[s]
        rescue += (not r) and d
        harm += r and (not d)
    ratio = (rescue / harm) if harm else float("inf")
    return rescue, harm, ratio


def ortho_diag(od: dict[int, dict]):
    cos_full, ratio_full, cos_act, ratio_act = [], [], [], []
    per_act_cos, per_act_ratio = [], []
    for sm in od.values():
        for step in sm.get("selector_trace", []):
            fv = step.get("orthogonal_full_vocab", {})
            av = step.get("orthogonal_action_vocab", {})
            if "cos_sim_raw" in av:
                cos_full.append(fv["cos_sim_raw"])
                ratio_full.append(fv["ortho_ratio"])
                cos_act.append(av["cos_sim_raw"])
                ratio_act.append(av["ortho_ratio"])
                if "per_action_cos_sim_raw" in av:
                    per_act_cos.append(np.asarray(av["per_action_cos_sim_raw"], dtype=float))
                    per_act_ratio.append(np.asarray(av["per_action_ortho_ratio"], dtype=float))

    def agg(x):
        x = np.asarray(x, dtype=float)
        return {"mean": float(x.mean()), "std": float(x.std()), "n": int(len(x))}

    return {
        "cos_sim_raw": {
            "full_vocab": agg(cos_full),
            "action_vocab": agg(cos_act),
        },
        "ortho_ratio": {
            "full_vocab": agg(ratio_full),
            "action_vocab": agg(ratio_act),
        },
        "per_action_cos_mean": (
            np.mean(per_act_cos, axis=0).tolist() if per_act_cos else None
        ),
        "per_action_ratio_mean": (
            np.mean(per_act_ratio, axis=0).tolist() if per_act_ratio else None
        ),
    }


def dual_only_rescue(od: dict[int, dict], ref: dict[int, dict[str, bool | None]], seeds: list[int]):
    out = []
    for s in sorted(seeds):
        r = ref[s]
        if (
            od[s]["success"]
            and r["vanilla"] is False
            and r["attn_uniform"] is False
            and r["attn_semantic"] is False
        ):
            out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--tasks",
        default=",".join(TASKS),
        help="comma-separated task names to analyze (default: all 9)",
    )
    args = ap.parse_args()
    selected = [t for t in args.tasks.split(",") if t]
    if any(t not in TASKS for t in selected):
        raise SystemExit(f"Unknown task in --tasks; must be one of {TASKS}")

    report: dict = {"tasks": {}}
    for task in selected:
        od = load_od(task)
        ref = load_ref(task)
        seeds = sorted(od.keys())
        od_succ = {s: bool(od[s]["success"]) for s in seeds}

        sr = {arm: None for arm in REF_ARMS}
        for arm in REF_ARMS:
            vals = [ref[s][arm] for s in seeds]
            if any(v is None for v in vals):
                sr[arm] = None
            else:
                sr[arm] = float(np.mean([1.0 if v else 0.0 for v in vals]))
        sr["orthogonal_dual"] = float(np.mean([1.0 if od_succ[s] else 0.0 for s in seeds]))

        pairwise = {}
        for arm in ("attn_uniform", "attn_semantic"):
            ref_succ = {s: bool(ref[s][arm]) for s in seeds}
            rescue, harm, ratio = rescue_harm(od_succ, ref_succ, seeds)
            obs, lo, hi = paired_bootstrap(
                [od_succ[s] for s in seeds], [ref_succ[s] for s in seeds]
            )
            pairwise[arm] = {
                "rescue": rescue,
                "harm": harm,
                "ratio": ratio,
                "mcnemar_p": mcnemar_exact(rescue, harm),
                "delta_sr": obs,
                "ci95": [lo, hi],
            }

        diag = ortho_diag(od)

        task_entry = {
            "n_seeds": len(seeds),
            "seeds": seeds,
            "sr": sr,
            "pairwise": pairwise,
            "diag": diag,
        }
        report["tasks"][task] = task_entry

    # ---- Coke dual-only rescue (only when coke is in the selected set) ----
    if "google_robot_pick_coke_can" in report["tasks"]:
        coke = report["tasks"]["google_robot_pick_coke_can"]
        coke_od = load_od("google_robot_pick_coke_can")
        coke_ref = load_ref("google_robot_pick_coke_can")
        coke_seeds = coke["seeds"]
        dor = dual_only_rescue(coke_od, coke_ref, coke_seeds)
        report["coke_dual_only_rescue"] = {"count": len(dor), "seeds": dor}

    # ---- print report ----
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
