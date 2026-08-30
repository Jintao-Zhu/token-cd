"""Contrastive-Evidence Gating offline diagnosis.

Fixed config: K=8, L8-15, lambda0=0.5 (single window, NOT mixing the 9-window sweep).
For each episode's FIRST vanilla->CD action flip, compute the flip-strength kappa:

    v      = argmax_a z+(a)            # vanilla (clean) action
    c      = argmax_a [z+(a) + lam0*r(a)],  r = z+ - z-    # CD action
    D+     = z+(v) - z+(c)              # vanilla resistance (>=0)
    Dr     = r(c) - r(v)                # semantic residual support (>0 at a flip)
    lam*   = D+ / (Dr + eps)            # minimum lambda to overturn vanilla
    kappa  = lam0 * Dr / (D+ + eps)     # flip strength; kappa>1 for any realized flip

Then test whether kappa separates rescue (vanilla fail -> CD success) from
harm (vanilla success -> CD fail), and run a fixed-threshold replay.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

LAMBDA0 = 0.5
EPS = 1e-6
ARTIFACT = "artifacts/attn_semantic_layer_window_sweep_v1"
ARM = "L8-15"
TASKS = ("google_robot_move_near", "google_robot_pick_coke_can", "google_robot_close_drawer")
SEEDS = range(100, 200)
GAMMAS = (1.1, 1.25, 1.5)


def load_success(task: str, arm: str, seed: int):
    p = f"{ARTIFACT}/episodes/{task}/{arm}/episode_{seed:03d}_summary.json"
    if not os.path.exists(p):
        return None
    return bool(json.load(open(p))["success"])


def load_arrays(task: str, arm: str, seed: int):
    p = f"{ARTIFACT}/episodes/{task}/{arm}/episode_{seed:03d}_arrays.npz"
    if not os.path.exists(p):
        return None
    return np.load(p)


def first_flip(zp: np.ndarray, zn: np.ndarray):
    """Return (t, q, v, c) of the first clean-vs-CD divergence.

    CD final = (1+lam) z+ - lam z- for tokens 0..5, z+ for token 6 (gripper).
    """
    zp = zp.astype(np.float32)
    zn = zn.astype(np.float32)
    wv = zp.argmax(-1)
    final = (1 + LAMBDA0) * zp - LAMBDA0 * zn
    final[:, 6] = zp[:, 6]
    wc = final.argmax(-1)
    T = min(zp.shape[0], zn.shape[0])
    for t in range(T):
        idx = np.flatnonzero(wv[t] != wc[t])
        if len(idx):
            q = int(idx[0])
            return t, q, int(wv[t, q]), int(wc[t, q])
    return None


def auc(rescue_k, harm_k):
    rescue_k = np.asarray(rescue_k, dtype=np.float64)
    harm_k = np.asarray(harm_k, dtype=np.float64)
    if len(rescue_k) == 0 or len(harm_k) == 0:
        return float("nan")
    total = 0.0
    for k in rescue_k:
        total += float((harm_k < k).sum()) + 0.5 * float((harm_k == k).sum())
    return total / (len(rescue_k) * len(harm_k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=ARTIFACT)
    args = ap.parse_args()
    os.chdir(os.path.expanduser("~/zjt_ws/token-cd"))

    print("=" * 88)
    print(f"Contrastive-Evidence Gating  |  K=8, {ARM}, lambda0={LAMBDA0}")
    print("=" * 88)

    table_rows = []
    for task in TASKS:
        rescue = {"kappa": [], "lam": [], "dp": [], "dr": [], "seed": []}
        harm = {"kappa": [], "lam": [], "dp": [], "dr": [], "seed": []}
        for seed in SEEDS:
            sv = load_success(task, "vanilla", seed)
            sc = load_success(task, ARM, seed)
            if sv is None or sc is None or sv == sc:
                continue  # neutral (both pass / both fail) not informative here
            dc = load_arrays(task, ARM, seed)
            if dc is None:
                continue
            zp = dc["positive_logits"]
            zn = dc["negative_logits"]
            f = first_flip(zp, zn)
            if f is None:
                continue
            t, q, v, c = f
            z = zp[t, q].astype(np.float32)
            r = (zp[t, q] - zn[t, q]).astype(np.float32)
            dp = float(z[v] - z[c])
            dr = float(r[c] - r[v])
            lam = dp / (dr + EPS)
            kappa = LAMBDA0 * dr / (dp + EPS)
            row = (rescue if (not sv and sc) else harm)
            row["kappa"].append(kappa)
            row["lam"].append(lam)
            row["dp"].append(dp)
            row["dr"].append(dr)
            row["seed"].append(seed)

        rk = np.array(rescue["kappa"]); hk = np.array(harm["kappa"])
        n_r, n_h = len(rk), len(hk)

        print(f"\n### {task}")
        for name, d in (("RESCUE", rescue), ("HARM", harm)):
            k = np.array(d["kappa"])
            if len(k) == 0:
                print(f"  {name}: n=0")
                continue
            print(f"  {name}: n={len(k)}")
            print(f"    kappa  mean={k.mean():.3f} med={np.median(k):.3f} "
                  f"q25={np.percentile(k,25):.3f} q75={np.percentile(k,75):.3f}")
            print(f"           <1.1:{ (k<1.1).mean():.0%}  >=1.25:{(k>=1.25).mean():.0%}  >=1.5:{(k>=1.5).mean():.0%}")
            lam = np.array(d["lam"])
            print(f"    lam*   med={np.median(lam):.3f}  (min lambda to flip)")
            print(f"    D+     med={np.median(d['dp']):.3f}  Dr med={np.median(d['dr']):.3f}")

        a = auc(rk, hk)
        print(f"  AUC(rescue>harm) = {a:.3f}   rescue_kappa>harm_kappa? {np.median(rk) > np.median(hk)}")
        table_rows.append((task, n_r, n_h, np.median(rk), np.median(hk), a))

        # threshold replay
        print(f"  Threshold replay (orig R/H = {n_r}/{n_h}):")
        for g in GAMMAS:
            rr = int((rk >= g).sum()); hh = int((hk >= g).sum())
            ratio = rr / hh if hh else float("inf")
            print(f"    kappa>={g}:  R {n_r}->{rr} ({rr/max(1,n_r):.0%})  H {n_h}->{hh} ({hh/max(1,n_h):.0%})  -> R/H = {rr}/{hh} = {ratio:.2f}")

    print("\n" + "=" * 88)
    print(f"{'Task':<28} {'Rescue k med':>13} {'Harm k med':>11} {'AUC':>6} {'R>H?':>5}")
    for task, n_r, n_h, rm, hm, a in table_rows:
        print(f"{task:<28} {rm:>13.3f} {hm:>11.3f} {a:>6.3f} {str(rm > hm):>5}")
    print("=" * 88)


if __name__ == "__main__":
    main()
