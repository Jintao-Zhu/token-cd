"""SCR-CD Phase 4 (full-coverage analysis): semantic_recon_k8_m10 over seeds 0-399.

Three parts:
  1. CORE tasks (move_near / close_drawer / pick_coke_can): 3-way paired
     recon-vs-vanilla / recon-vs-attn over 0-399, split into 4 blocks
     (0-99 / 100-199 / 200-299 / 300-399) + full 0-399. This EXTENDS Phase 2
     (which only covered 0-199) with the two new blocks.
  2. OTHER 6 tasks (open_drawer / place_apple / carrot_on_plate /
     put_eggplant_in_basket / spoon_on_towel / stack_cube): 5-arm paired
     analysis at seeds 300-399 (the ONLY block where vanilla/attn/merge/random
     baselines exist). recon is the reference; compare vs vanilla / attn /
     merge / random_recon.
  3. OTHER 6 tasks: single-arm recon success rate over 0-299 (backfill block,
     no baseline) + 300-399, for absolute SR coverage.

Arm aliasing (explicit, so comparisons are clean):
  attn_semantic_k8 (0-99) == L8-15 (100-199) == semantic_attn_k8_l8_15 (canonical)
  All: K=8 semantic selector, layers 8-15, lambda=0.5, semantic_hard, mask -1e4.

Fail-closed pairing: a seed enters a comparison only if every involved arm
agrees on BOTH initial_state_sha256 AND canonical_snapshot_sha256. (Cross-process
SAPIEN float non-determinism otherwise: recon runs in a separate process from the
baselines, so ~4% of seeds get a non-bit-identical env.reset.)

Output: artifacts/semantic_recon_0_399_full_v1/analysis.json (+ console report).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

CORE_TASKS = (
    "google_robot_move_near",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
)
OTHER_TASKS = (
    "google_robot_open_drawer",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_stack_cube",
)
ALL_TASKS = CORE_TASKS + OTHER_TASKS

A_K8K16 = "artifacts/attn_semantic_k8k16_100seed_v1"
A_LAYER = "artifacts/attn_semantic_layer_window_sweep_v1"
A_GMERGE = "artifacts/attn_global_merge_v1"
A_SMERGE = "artifacts/attn_semantic_merge_k8_v1"
A_PH1 = "artifacts/semantic_recon_k8_m10_v1"
A_PH2 = "artifacts/semantic_recon_0_199_core3_replication_v1"
A_PH3 = "artifacts/semantic_recon_k8_m10_0_299_backfill_v1"

ARM_VANILLA = "vanilla"
ARM_ATTN = "semantic_attn_k8_l8_15"  # canonical name for attn_semantic_k8 / L8-15


def resolve_source(task: str, arm: str, seed: int):
    """Return (artifact, subdir) for a (task, canonical_arm, seed). arm in
    {vanilla, semantic_attn_k8_l8_15, semantic_recon_k8_m10, ...}."""
    if arm == "vanilla":
        if task in CORE_TASKS:
            if seed <= 99:
                # move_near 0-99 vanilla is the Phase-2 backfill; others in k8k16.
                if task == "google_robot_move_near":
                    return A_PH2, "vanilla"
                return A_K8K16, "vanilla"
            if seed <= 199:
                return A_LAYER, "vanilla"
            if seed <= 299:
                return A_GMERGE, "vanilla"
            return A_PH1, "vanilla"
        else:  # OTHER_TASKS: only 300-399
            return A_PH1, "vanilla"
    if arm == ARM_ATTN:
        if task in CORE_TASKS:
            if seed <= 99:
                if task == "google_robot_move_near":
                    return A_PH2, "semantic_attn_k8_l8_15"
                return A_K8K16, "attn_semantic_k8"
            if seed <= 199:
                return A_LAYER, "L8-15"
            if seed <= 299:
                return A_SMERGE, "semantic_attn_k8_l8_15"
            return A_PH1, "semantic_attn_k8_l8_15"
        else:
            return A_PH1, "semantic_attn_k8_l8_15"
    if arm == "semantic_recon_k8_m10":
        if task in CORE_TASKS:
            if seed <= 199:
                return A_PH2, "semantic_recon_k8_m10"
            if seed <= 299:
                return A_PH3, "semantic_recon_k8_m10"
            return A_PH1, "semantic_recon_k8_m10"
        else:
            if seed <= 299:
                return A_PH3, "semantic_recon_k8_m10"
            return A_PH1, "semantic_recon_k8_m10"
    # 5-arm extras (Phase 1, same process)
    return A_PH1, arm


@lru_cache(maxsize=None)
def load_dir(artifact: str, task: str, subdir: str) -> dict[int, dict]:
    d = Path(artifact) / "episodes" / task / subdir
    out: dict[int, dict] = {}
    if not d.is_dir():
        return out
    for f in sorted(d.glob("episode_*_summary.json")):
        seed = int(f.name.split("_")[1])
        try:
            out[seed] = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
    return out


def load_records(tasks, arms, lo, hi) -> dict[tuple[str, str, int], dict]:
    rec: dict[tuple[str, str, int], dict] = {}
    for task in tasks:
        for arm in arms:
            for seed in range(lo, hi + 1):
                art, sub = resolve_source(task, arm, seed)
                s = load_dir(art, task, sub).get(seed)
                if s is not None:
                    rec[(task, arm, seed)] = s
    return rec


def seed_boot_ci(deltas: np.ndarray, n: int = 4000, seed: int = 731902):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(deltas), len(deltas))
        vals.append(float(deltas[idx].mean()))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def paired(rec, tasks, arm_x, arm_y, lo, hi):
    """Paired stats of arm_y vs arm_x over [lo,hi], per task + pooled."""
    per_seed: dict[tuple[str, int], dict[str, int]] = defaultdict(dict)
    for (task, a, seed), s in rec.items():
        if a in (arm_x, arm_y) and lo <= seed <= hi:
            per_seed[(task, seed)][a] = int(s["success"])

    out, all_d, all_xs, all_ys, all_n01, all_n10 = {}, [], [], [], 0, 0
    for task in tasks:
        xs, ys, n01, n10 = [], [], 0, 0
        for (t, seed), arms in sorted(per_seed.items()):
            if t != task or arm_x not in arms or arm_y not in arms:
                continue
            x, y = arms[arm_x], arms[arm_y]
            xs.append(x); ys.append(y)
            if x == 0 and y == 1:
                n01 += 1
            elif x == 1 and y == 0:
                n10 += 1
        d = np.array([y - x for x, y in zip(xs, ys)], dtype=int)
        if len(d) == 0:
            out[task] = {"n_pairs": 0}
            continue
        disc = n01 + n10
        out[task] = {
            "n_pairs": int(len(d)),
            "sr_x": float(np.mean(xs)),
            "sr_y": float(np.mean(ys)),
            "delta": float(d.mean()),
            "rescued": n01, "harmed": n10, "net": n01 - n10,
            "mcnemar_p": float(binomtest(n01, disc).pvalue) if disc else 1.0,
            "ci95": list(seed_boot_ci(d.astype(float))),
        }
        all_d.append(d); all_xs.extend(xs); all_ys.extend(ys); all_n01 += n01; all_n10 += n10
    if all_d:
        pd_ = np.concatenate(all_d); disc = all_n01 + all_n10
        out["POOLED"] = {
            "n_pairs": int(len(pd_)), "sr_x": float(np.mean(all_xs)), "sr_y": float(np.mean(all_ys)),
            "delta": float(pd_.mean()),
            "rescued": all_n01, "harmed": all_n10, "net": all_n01 - all_n10,
            "mcnemar_p": float(binomtest(all_n01, disc).pvalue) if disc else 1.0,
            "ci95": list(seed_boot_ci(pd_.astype(float))),
        }
    return out


def hash_valid(rec, tasks, arms, lo, hi):
    """Seeds where all `arms` agree on both hash fields. Returns (valid, fail)."""
    per_seed: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for (task, a, seed), s in rec.items():
        if a in arms and lo <= seed <= hi:
            per_seed[(task, seed)][a] = s
    valid, fail = [], []
    for (task, seed), am in sorted(per_seed.items()):
        if task not in tasks:
            continue
        if set(am) != set(arms):
            continue
        hs = {a: am[a].get("initial_state_sha256") for a in arms}
        cs = {a: am[a].get("canonical_snapshot_sha256") for a in arms}
        if len(set(hs.values())) != 1 or len(set(cs.values())) != 1:
            fail.append((task, seed))
        else:
            valid.append((task, seed))
    return valid, fail


def run_paired_suite(rec, tasks, blocks, y_arm, x_arms):
    """For each block, hash-valid filter then paired y vs each x."""
    suite = {}
    for bname, (lo, hi) in blocks.items():
        arms_needed = {y_arm} | set(x_arms)
        valid, fail = hash_valid(rec, tasks, arms_needed, lo, hi)
        suite[bname] = {"lo": lo, "hi": hi, "valid_pairs": len(valid),
                        "hash_failures": fail}
        for x in x_arms:
            suite[bname][f"{y_arm}_vs_{x}"] = paired(rec, tasks, x, y_arm, lo, hi)
    return suite


def single_arm_sr(tasks, arm, lo, hi):
    res = {}
    for t in tasks:
        succ = []
        for seed in range(lo, hi + 1):
            art, sub = resolve_source(t, arm, seed)
            s = load_dir(art, t, sub).get(seed)
            if s is not None:
                succ.append(int(s["success"]))
        res[t] = {"n": len(succ), "sr": float(np.mean(succ)) if succ else 0.0}
    return res


def fmt_pair(p):
    if not p.get("n_pairs"):
        return "n=0"
    return (f"n={p['n_pairs']:3d} SRx={p['sr_x']:.3f} SRy={p['sr_y']:.3f} "
            f"dSR={p['delta']:+.3f} R={p['rescued']} H={p['harmed']} net={p['net']:+d} "
            f"p={p['mcnemar_p']:.3g} CI=[{p['ci95'][0]:+.3f},{p['ci95'][1]:+.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("artifacts/semantic_recon_0_399_full_v1/analysis.json"))
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    result = {"parts": {}}

    # ---------- Part 1: CORE 0-399 ----------
    core_arms = [ARM_VANILLA, ARM_ATTN, "semantic_recon_k8_m10"]
    core_rec = load_records(CORE_TASKS, core_arms, 0, 399)
    blocks = {"0-99": (0, 99), "100-199": (100, 199),
              "200-299": (200, 299), "300-399": (300, 399), "0-399": (0, 399)}
    part1 = {}
    for bname, (lo, hi) in blocks.items():
        valid, fail = hash_valid(core_rec, CORE_TASKS, core_arms, lo, hi)
        part1[bname] = {
            "valid_pairs": len(valid), "hash_failures": fail,
            "recon_vs_vanilla": paired(core_rec, CORE_TASKS, ARM_VANILLA, "semantic_recon_k8_m10", lo, hi),
            "recon_vs_attn": paired(core_rec, CORE_TASKS, ARM_ATTN, "semantic_recon_k8_m10", lo, hi),
        }
    result["parts"]["core_0_399"] = part1

    # ---------- Part 2: OTHER 6 tasks, 300-399, 5-arm ----------
    other_arms = [ARM_VANILLA, ARM_ATTN, "semantic_merge_k8_eta100",
                  "random_recon_k8_m10", "semantic_recon_k8_m10"]
    other_rec = load_records(OTHER_TASKS, other_arms, 300, 399)
    valid, fail = hash_valid(other_rec, OTHER_TASKS, other_arms, 300, 399)
    part2 = {"valid_pairs": len(valid), "hash_failures": fail}
    for x in [ARM_VANILLA, ARM_ATTN, "semantic_merge_k8_eta100", "random_recon_k8_m10"]:
        part2[f"recon_vs_{x}"] = paired(other_rec, OTHER_TASKS, x, "semantic_recon_k8_m10", 300, 399)
    result["parts"]["other_300_399"] = part2

    # ---------- Part 3: OTHER 6 tasks, single-arm recon SR ----------
    part3 = {}
    for label, (lo, hi) in {"0-299": (0, 299), "300-399": (300, 399), "0-399": (0, 399)}.items():
        part3[label] = single_arm_sr(OTHER_TASKS, "semantic_recon_k8_m10", lo, hi)
    result["parts"]["other_single_arm_sr"] = part3

    # ================= console report =================
    print("=" * 100)
    print("SCR-CD Phase 4: semantic_recon_k8_m10 full-coverage analysis (0-399)")
    print("arm aliasing: attn_semantic_k8 == L8-15 == semantic_attn_k8_l8_15")
    print("=" * 100)

    print("\n### PART 1 — CORE tasks (move_near / close_drawer / pick_coke_can), 0-399")
    for bname in blocks:
        b = part1[bname]
        print(f"\n--- block {bname}: valid_pairs={b['valid_pairs']}, hash_fail={len(b['hash_failures'])} ---")
        for task in CORE_TASKS:
            rv = b["recon_vs_vanilla"].get(task, {})
            ra = b["recon_vs_attn"].get(task, {})
            if not rv.get("n_pairs") and not ra.get("n_pairs"):
                print(f"  {task:45s} n=0")
                continue
            print(f"  {task:45s}")
            if rv.get("n_pairs"):
                print(f"      vs vanilla: {fmt_pair(rv)}")
            if ra.get("n_pairs"):
                print(f"      vs attn   : {fmt_pair(ra)}")
        pv = b["recon_vs_vanilla"].get("POOLED", {})
        pa = b["recon_vs_attn"].get("POOLED", {})
        if pv.get("n_pairs"):
            print(f"  [POOLED n={pv['n_pairs']}] recon-vanilla {fmt_pair(pv)}")
        if pa.get("n_pairs"):
            print(f"  [POOLED n={pa['n_pairs']}] recon-attn    {fmt_pair(pa)}")

    print("\n\n### PART 2 — OTHER 6 tasks, 300-399 (5-arm, hash-verified)")
    print(f"valid_pairs={part2['valid_pairs']}, hash_fail={len(part2['hash_failures'])}")
    for x in [ARM_VANILLA, ARM_ATTN, "semantic_merge_k8_eta100", "random_recon_k8_m10"]:
        print(f"\n  recon vs {x}:")
        suite = part2[f"recon_vs_{x}"]
        for task in OTHER_TASKS:
            p = suite.get(task, {})
            print(f"    {task:48s} {fmt_pair(p)}")
        pp = suite.get("POOLED", {})
        if pp.get("n_pairs"):
            print(f"    [POOLED n={pp['n_pairs']}] {fmt_pair(pp)}")

    print("\n\n### PART 3 — OTHER 6 tasks, single-arm recon SR (absolute)")
    for label in ["0-299", "300-399", "0-399"]:
        print(f"\n  {label}:")
        for task in OTHER_TASKS:
            r = part3[label][task]
            print(f"    {task:48s} n={r['n']:3d} SR={r['sr']:.3f}")

    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\n[wrote] {a.out}")


if __name__ == "__main__":
    main()
