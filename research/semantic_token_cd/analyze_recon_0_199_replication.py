"""SCR-CD Phase 2 (replication): 3-way paired analysis, seeds 0-199, split by block.

Compares semantic_recon_k8_m10 (new, this run) against EXISTING vanilla + semantic
attn baselines over the SAME initial-state range, split into 0-99 / 100-199 / 0-199.

Data sources (arm-name aliasing is explicit so the comparison is clean):
  recon      -> artifacts/semantic_recon_0_199_core3_replication_v1/<task>/semantic_recon_k8_m10
  vanilla/attn 0-99:
      close_drawer, pick_coke_can -> artifacts/attn_semantic_k8k16_100seed_v1/<task>/{vanilla,attn_semantic_k8}
      move_near                   -> artifacts/semantic_recon_0_199_core3_replication_v1/move_near/{vanilla,semantic_attn_k8_l8_15} (backfill)
  vanilla/attn 100-199:
      all 3 tasks                 -> artifacts/attn_semantic_layer_window_sweep_v1/<task>/{vanilla,L8-15}

Aliasing: attn_semantic_k8 == L8-15 == semantic_attn_k8_l8_15 (all K=8, layers 8-15,
lambda 0.5, semantic_hard). Fail-closed pairing on initial_state_sha256 (and
canonical_snapshot_sha256) — a seed enters a block only if all 3 arms agree.

Output: per task x block {0-99, 100-199, 0-199}, for recon-vs-vanilla AND
recon-vs-attn: valid n, SR, delta SR, rescue/harm/net, exact McNemar, paired 95% CI.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

TASKS = (
    "google_robot_move_near",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
)
CANONICAL = ("vanilla", "attn", "recon")
RECON_ART = "artifacts/semantic_recon_0_199_core3_replication_v1"
OLD_0_99 = "artifacts/attn_semantic_k8k16_100seed_v1"
OLD_100_199 = "artifacts/attn_semantic_layer_window_sweep_v1"

# (canonical_arm) -> list of (artifact, task, arm_subdir)
# For move_near the 0-99 vanilla/attn live in the new artifact (backfill).
SOURCES = {
    "vanilla": [
        (OLD_0_99, "google_robot_close_drawer", "vanilla"),
        (OLD_0_99, "google_robot_pick_coke_can", "vanilla"),
        (OLD_100_199, "google_robot_move_near", "vanilla"),
        (OLD_100_199, "google_robot_close_drawer", "vanilla"),
        (OLD_100_199, "google_robot_pick_coke_can", "vanilla"),
        (RECON_ART, "google_robot_move_near", "vanilla"),  # backfill 0-99
    ],
    "attn": [
        (OLD_0_99, "google_robot_close_drawer", "attn_semantic_k8"),
        (OLD_0_99, "google_robot_pick_coke_can", "attn_semantic_k8"),
        (OLD_100_199, "google_robot_move_near", "L8-15"),
        (OLD_100_199, "google_robot_close_drawer", "L8-15"),
        (OLD_100_199, "google_robot_pick_coke_can", "L8-15"),
        (RECON_ART, "google_robot_move_near", "semantic_attn_k8_l8_15"),  # backfill 0-99
    ],
    "recon": [
        (RECON_ART, t, "semantic_recon_k8_m10") for t in TASKS
    ],
}


def load_dir(artifact: str, task: str, arm_subdir: str) -> dict[int, dict]:
    d = Path(artifact) / "episodes" / task / arm_subdir
    out: dict[int, dict] = {}
    if not d.is_dir():
        return out
    for f in sorted(d.glob("episode_*_summary.json")):
        seed = int(f.name.split("_")[1])
        try:
            out[seed] = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
    return out


def load_records() -> tuple[dict, dict]:
    """records[(task, canonical_arm, seed)] = summary; src_meta for provenance."""
    rec: dict[tuple[str, str, int], dict] = {}
    for arm in CANONICAL:
        for artifact, task, subdir in SOURCES[arm]:
            for seed, s in load_dir(artifact, task, subdir).items():
                rec[(task, arm, seed)] = s
    return rec, {}


def seed_boot_ci(deltas: np.ndarray, n: int = 4000, seed: int = 731902):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(deltas), len(deltas))
        vals.append(float(deltas[idx].mean()))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def paired_analysis(rec: dict, block: tuple[int, int], arm_x: str, arm_y: str):
    """Paired stats of arm_y vs arm_x over a seed block (per task + pooled)."""
    lo, hi = block
    by_task: dict[str, list[tuple[int, int]]] = defaultdict(list)  # (y_succ, x_succ)
    for (task, a, seed), s in rec.items():
        if a not in (arm_x, arm_y):
            continue
        if not (lo <= seed <= hi):
            continue
        by_task[task].append((seed, a, int(s["success"])))

    # build per-seed paired (y - x)
    out = {}
    all_d = []
    all_n01 = all_n10 = 0
    for task in TASKS:
        per_seed: dict[int, dict[str, int]] = defaultdict(dict)
        for seed, a, succ in by_task[task]:
            per_seed[seed][a] = succ
        d = []
        n01 = n10 = 0
        for seed in sorted(per_seed):
            if arm_x not in per_seed[seed] or arm_y not in per_seed[seed]:
                continue
            x, y = per_seed[seed][arm_x], per_seed[seed][arm_y]
            d.append(y - x)
            if x == 0 and y == 1:
                n01 += 1
            elif x == 1 and y == 0:
                n10 += 1
        d = np.array(d, dtype=int)
        if len(d) == 0:
            out[task] = {"n_pairs": 0}
            continue
        disc = n01 + n10
        mcn = float(binomtest(n01, disc).pvalue) if disc else 1.0
        ci = seed_boot_ci(d.astype(float))
        out[task] = {
            "n_pairs": int(len(d)),
            "sr_x": float(np.mean([per_seed[s][arm_x] for s in sorted(per_seed)
                                   if arm_x in per_seed[s] and arm_y in per_seed[s]])),
            "sr_y": float(np.mean([per_seed[s][arm_y] for s in sorted(per_seed)
                                   if arm_x in per_seed[s] and arm_y in per_seed[s]])),
            "delta": float(d.mean()),
            "rescued": n01,
            "harmed": n10,
            "net": n01 - n10,
            "mcnemar_p": mcn,
            "ci95": list(ci),
        }
        all_d.append(d)
        all_n01 += n01
        all_n10 += n10
    if all_d:
        pooled_d = np.concatenate(all_d)
        disc = all_n01 + all_n10
        out["POOLED"] = {
            "n_pairs": int(len(pooled_d)),
            "delta": float(pooled_d.mean()),
            "rescued": all_n01,
            "harmed": all_n10,
            "net": all_n01 - all_n10,
            "mcnemar_p": float(binomtest(all_n01, disc).pvalue) if disc else 1.0,
            "ci95": list(seed_boot_ci(pooled_d.astype(float))),
        }
    return out


def hash_check(rec: dict, block: tuple[int, int]) -> tuple[list, list]:
    """Valid seeds where all 3 arms agree on initial_state_sha256 (and canonical)."""
    lo, hi = block
    per_seed: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for (task, a, seed), s in rec.items():
        if lo <= seed <= hi:
            per_seed[(task, seed)][a] = s
    valid, fail = [], []
    for (task, seed), arms in sorted(per_seed.items()):
        if set(arms) != set(CANONICAL):
            continue  # incomplete -> skip (not a hash failure)
        hs = {a: arms[a].get("initial_state_sha256") for a in CANONICAL}
        cs = {a: arms[a].get("canonical_snapshot_sha256") for a in CANONICAL}
        if len(set(hs.values())) != 1 or len(set(cs.values())) != 1:
            fail.append((task, seed, hs, cs))
        else:
            valid.append((task, seed))
    return valid, fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None, help="JSON output path")
    a = ap.parse_args()

    rec, _ = load_records()
    # inventory
    inv = defaultdict(lambda: defaultdict(int))
    for (task, arm, seed) in rec:
        inv[task][arm] += 1

    blocks = {"0-99": (0, 99), "100-199": (100, 199), "0-199": (0, 199)}
    result = {"inventory": {t: dict(inv[t]) for t in TASKS}, "blocks": {}}

    for bname, block in blocks.items():
        valid, fail = hash_check(rec, block)
        rv = paired_analysis(rec, block, "vanilla", "recon")
        ra = paired_analysis(rec, block, "attn", "recon")
        result["blocks"][bname] = {
            "valid_pairs": len(valid),
            "hash_failures": [{"task": t, "seed": s} for (t, s, _, _) in fail],
            "recon_vs_vanilla": rv,
            "recon_vs_attn": ra,
        }

    # ---------- console report ----------
    print("=== SCR-CD Phase 2 replication: recon vs vanilla/attn, 0-199 (hash-verified) ===")
    print("arm aliasing: attn_semantic_k8 == L8-15 == semantic_attn_k8_l8_15")
    print("inventory (episodes loaded):")
    for t in TASKS:
        print(f"  {t:45s} " + " ".join(f"{arm}={inv[t].get(arm,0)}" for arm in CANONICAL))
    print()
    for bname, block in blocks.items():
        b = result["blocks"][bname]
        print(f"--- block {bname} (valid pairs={b['valid_pairs']}, hash_fail={len(b['hash_failures'])}) ---")
        for task in TASKS:
            rv = b["recon_vs_vanilla"].get(task, {})
            ra = b["recon_vs_attn"].get(task, {})
            if not rv.get("n_pairs") and not ra.get("n_pairs"):
                print(f"  {task:45s} n=0 (no paired data)")
                continue
            print(f"  {task:45s} n={rv.get('n_pairs', ra.get('n_pairs'))}")
            if rv.get("n_pairs"):
                print(f"      recon vs vanilla: SR van={rv['sr_x']:.2f} rec={rv['sr_y']:.2f} "
                      f"dSR={rv['delta']:+.3f}  R={rv['rescued']} H={rv['harmed']} net={rv['net']:+d} "
                      f"mcnemar_p={rv['mcnemar_p']:.3g}  CI95=[{rv['ci95'][0]:+.3f},{rv['ci95'][1]:+.3f}]")
            if ra.get("n_pairs"):
                print(f"      recon vs attn   : SR attn={ra['sr_x']:.2f} rec={ra['sr_y']:.2f} "
                      f"dSR={ra['delta']:+.3f}  R={ra['rescued']} H={ra['harmed']} net={ra['net']:+d} "
                      f"mcnemar_p={ra['mcnemar_p']:.3g}  CI95=[{ra['ci95'][0]:+.3f},{ra['ci95'][1]:+.3f}]")
        # pooled
        pv = b["recon_vs_vanilla"].get("POOLED", {})
        pa = b["recon_vs_attn"].get("POOLED", {})
        if pv.get("n_pairs"):
            print(f"  [POOLED n={pv['n_pairs']}] recon-vanilla dSR={pv['delta']:+.3f} "
                  f"R={pv['rescued']} H={pv['harmed']} net={pv['net']:+d} p={pv['mcnemar_p']:.3g}")
        if pa.get("n_pairs"):
            print(f"  [POOLED n={pa['n_pairs']}] recon-attn    dSR={pa['delta']:+.3f} "
                  f"R={pa['rescued']} H={pa['harmed']} net={pa['net']:+d} p={pa['mcnemar_p']:.3g}")
        print()

    if a.out:
        a.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[wrote] {a.out}")


if __name__ == "__main__":
    main()
